"""A focused, Mojo-accelerated subset of the :mod:`pycollada` API.

It reads DAE geometry, skin controllers, and visual scenes.  XML is parsed in
Python; dense indexed expansion, transform products, and skinning run in Mojo.
"""

from __future__ import annotations

import io
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from .controller import Skin
from .geometry import Geometry, InputList, Polylist, TriangleSet
from .scene import ControllerNode, GeometryNode, Node, NodeNode, Scene
from .source import FloatSource, NameSource

__all__ = [
    "Collada", "ColladaError", "DaeMalformedError", "Geometry", "InputList",
    "FloatSource", "NameSource", "TriangleSet", "Polylist", "Skin", "Scene", "Node",
    "GeometryNode", "ControllerNode",
]


class ColladaError(Exception):
    pass


class DaeMalformedError(ColladaError):
    pass


class Asset:
    def __init__(self, unitmeter=1.0, upaxis="Y_UP"):
        self.unitmeter = unitmeter
        self.upaxis = upaxis


def _tag(node):
    return node.tag.rsplit("}", 1)[-1]


def _children(node, name):
    namespaced = "}" + name
    return [child for child in node if child.tag == name or child.tag.endswith(namespaced)]


def _child(node, name):
    namespaced = "}" + name
    for child in node:
        if child.tag == name or child.tag.endswith(namespaced):
            return child
    return None


def _numbers(node, dtype=float):
    text = "" if node is None else (node.text or "").strip()
    if not text:
        return []
    if len(text) >= 256:
        return np.fromstring(text, dtype=dtype, sep=" ")
    return [dtype(value) for value in text.split()]


def _matrix(values):
    if len(values) != 16:
        raise DaeMalformedError("a COLLADA matrix must contain 16 values")
    return np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")


def _translation(values):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = values[:3]
    return result


def _scale(values):
    result = np.eye(4, dtype=np.float64)
    result[0, 0], result[1, 1], result[2, 2] = values[:3]
    return result


def _rotation(values):
    axis = np.asarray(values[:3], dtype=np.float64)
    length = np.linalg.norm(axis)
    if length == 0:
        return np.eye(4)
    axis /= length
    x, y, z = axis
    c, s = math.cos(math.radians(values[3])), math.sin(math.radians(values[3]))
    d = 1.0 - c
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c + x*x*d, x*y*d-z*s, x*z*d+y*s),
                      (y*x*d+z*s, c+y*y*d, y*z*d-x*s),
                      (z*x*d-y*s, z*y*d+x*s, c+z*z*d))
    return result


class Collada:
    def __init__(self, filename=None, ignore=None, aux_file_loader=None, zip_filename=None, validate_output=False):
        self.filename = filename
        self.ignore = ignore
        self.aux_file_loader = aux_file_loader
        self.zip_filename = zip_filename
        self.validate_output = validate_output
        self.assetInfo = Asset()
        self.geometries = []
        self.controllers = []
        self.scenes = []
        self.scene = None
        self._geometry_by_id = {}
        self._controller_by_id = {}
        if filename is not None:
            self.load(filename)

    def load(self, source):
        try:
            if hasattr(source, "read"):
                root = ET.parse(source).getroot()
            elif isinstance(source, (bytes, bytearray)):
                root = ET.fromstring(source)
            else:
                self.filename = os.fspath(source)
                root = ET.parse(self.filename).getroot()
        except (ET.ParseError, OSError) as exc:
            raise DaeMalformedError(str(exc)) from exc
        if _tag(root) != "COLLADA":
            raise DaeMalformedError("root element is not COLLADA")
        self._parse_asset(root)
        self._parse_geometries(root)
        self._parse_controllers(root)
        self._parse_scenes(root)
        return self

    def _parse_asset(self, root):
        asset = _child(root, "asset")
        if asset is None:
            return
        unit = _child(asset, "unit")
        up = _child(asset, "up_axis")
        self.assetInfo = Asset(float(unit.get("meter", "1")) if unit is not None else 1.0,
                               (up.text or "Y_UP").strip() if up is not None else "Y_UP")

    def _sources(self, container):
        result = {}
        for node in _children(container, "source"):
            array = _child(node, "float_array")
            names = _child(node, "Name_array")
            if names is None:
                names = _child(node, "IDREF_array")
            accessor = _child(_child(node, "technique_common"), "accessor")
            stride = int(accessor.get("stride", "1")) if accessor is not None else 1
            count = int(accessor.get("count", "0")) if accessor is not None else 0
            offset = int(accessor.get("offset", "0")) if accessor is not None else 0
            components = [param.get("name", "") for param in _children(accessor, "param")] if accessor is not None else []
            if array is not None:
                values = np.asarray(_numbers(array), dtype=np.float64)
                count = count or len(values) // stride
                if stride <= 0 or count < 0 or offset < 0 or len(values) < offset + count * stride:
                    raise DaeMalformedError("source accessor exceeds its float array")
                result[node.get("id")] = FloatSource(node.get("id"), values[offset:offset + count * stride].reshape(count, stride), components, node)
            elif names is not None:
                values = (names.text or "").split()
                if stride != 1 or offset < 0 or (count and len(values) < offset + count):
                    raise DaeMalformedError("unsupported or invalid name source accessor")
                result[node.get("id")] = NameSource(node.get("id"), values[offset:offset + count] if count else values[offset:], components, node)
        return result

    def _inputs(self, primitive, vertices):
        inputs = InputList()
        for node in _children(primitive, "input"):
            semantic, source = node.get("semantic"), node.get("source")
            if semantic == "VERTEX":
                vertex_inputs = vertices.get(source.lstrip("#"), [])
                position = next((value for value in vertex_inputs if value[0] == "POSITION"), None)
                if position is None:
                    raise DaeMalformedError("VERTEX input has no POSITION source")
                source = position[1]
            inputs.addInput(node.get("offset", "0"), semantic, source, node.get("set"))
        return inputs

    def _parse_geometries(self, root):
        library = _child(root, "library_geometries")
        if library is None:
            return
        for element in _children(library, "geometry"):
            mesh = _child(element, "mesh")
            if mesh is None:
                continue
            sources = self._sources(mesh)
            vertices = {}
            for vertex in _children(mesh, "vertices"):
                vertices[vertex.get("id")] = [(item.get("semantic"), item.get("source")) for item in _children(vertex, "input")]
            geometry = Geometry(self, element.get("id"), element.get("name"), sources, xmlnode=element)
            for primitive in mesh:
                kind = _tag(primitive)
                if kind not in {"triangles", "polylist"}:
                    if kind not in {"source", "vertices", "extra"}:
                        raise ColladaError(f"unsupported mesh primitive: {kind}")
                    continue
                inputs = self._inputs(primitive, vertices)
                width = max((item.offset for item in inputs.inputs), default=-1) + 1
                raw = np.asarray(_numbers(_child(primitive, "p"), int), dtype=np.int64)
                if width == 0 or len(raw) % width:
                    raise DaeMalformedError("primitive index stream does not match inputs")
                indices = raw.reshape((-1, width))
                if kind == "triangles":
                    if len(indices) % 3:
                        raise DaeMalformedError("triangle index count is not divisible by three")
                    geometry.createTriangleSet(indices, inputs, primitive.get("material"))
                else:
                    vcount = np.asarray(_numbers(_child(primitive, "vcount"), int), dtype=np.int64)
                    if int(vcount.sum()) != len(indices):
                        raise DaeMalformedError("polylist vcount does not match index stream")
                    geometry.createPolylist(indices, vcount, inputs, primitive.get("material"))
            self.geometries.append(geometry)
            self._geometry_by_id[geometry.id] = geometry

    def _parse_controllers(self, root):
        library = _child(root, "library_controllers")
        if library is None:
            return
        for node in _children(library, "controller"):
            skin = _child(node, "skin")
            if skin is None:
                continue
            sources = self._sources(skin)
            joints = _child(skin, "joints")
            weights_node = _child(skin, "vertex_weights")
            if joints is None or weights_node is None:
                raise DaeMalformedError("skin controller is missing joints or vertex_weights")
            joint_inputs = {item.get("semantic"): item.get("source").lstrip("#") for item in _children(joints, "input")}
            weight_inputs = _children(weights_node, "input")
            index_by_semantic = {item.get("semantic"): int(item.get("offset", "0")) for item in weight_inputs}
            if not {"JOINT", "WEIGHT"}.issubset(index_by_semantic) or "JOINT" not in joint_inputs or "INV_BIND_MATRIX" not in joint_inputs:
                raise DaeMalformedError("skin controller has incomplete joint or weight inputs")
            width = max(index_by_semantic.values()) + 1
            vcount = np.asarray(_numbers(_child(weights_node, "vcount"), int), dtype=np.int64)
            flat = np.asarray(_numbers(_child(weights_node, "v"), int), dtype=np.int64)
            if width <= 0 or len(flat) % width or len(flat) // width != int(vcount.sum()):
                raise DaeMalformedError("skin vertex weight stream does not match vcount")
            raw = flat.reshape((-1, width))
            bind = _child(skin, "bind_shape_matrix")
            inverse = sources[joint_inputs["INV_BIND_MATRIX"]].data.reshape((-1, 4, 4)).transpose((0, 2, 1))
            weight_source = sources[next(item.get("source").lstrip("#") for item in weight_inputs if item.get("semantic") == "WEIGHT")]
            geometry_id = skin.get("source").lstrip("#")
            controller = Skin(node.get("id"), geometry_id, sources[joint_inputs["JOINT"]], inverse,
                              weight_source.data,
                              vcount, raw[:, index_by_semantic["JOINT"]], raw[:, index_by_semantic["WEIGHT"]],
                              _matrix(_numbers(bind)) if bind is not None else None,
                              self._geometry_by_id.get(geometry_id), sources[joint_inputs["INV_BIND_MATRIX"]], weight_source, raw.reshape(-1), sources)
            self.controllers.append(controller)
            self._controller_by_id[controller.id] = controller

    def _parse_node(self, element, registry):
        transforms = []
        children = []
        node = Node(element.get("id"), children, transforms, element.get("name"), element.get("sid"), element)
        if node.id:
            registry[node.id] = node
        for child in element:
            kind = _tag(child)
            if kind == "matrix": transforms.append(_matrix(_numbers(child)))
            elif kind == "translate": transforms.append(_translation(_numbers(child)))
            elif kind == "scale": transforms.append(_scale(_numbers(child)))
            elif kind == "rotate": transforms.append(_rotation(_numbers(child)))
            elif kind == "node": children.append(self._parse_node(child, registry))
            elif kind == "instance_geometry": children.append(GeometryNode(self._geometry_by_id[child.get("url").lstrip("#")]))
            elif kind == "instance_controller":
                children.append(ControllerNode(self._controller_by_id[child.get("url").lstrip("#")], skeletons=[(value.text or "").lstrip("#") for value in _children(child, "skeleton")]))
            elif kind == "instance_node": children.append(("instance_node", child.get("url").lstrip("#")))
        return node

    def _resolve_instances(self, node, registry):
        node.children[:] = [NodeNode(registry[item[1]]) if isinstance(item, tuple) and item[0] == "instance_node" else item for item in node.children]
        for child in node.children:
            if isinstance(child, Node): self._resolve_instances(child, registry)

    def _parse_scenes(self, root):
        library = _child(root, "library_visual_scenes")
        if library is None:
            return
        registry = {}
        for element in _children(library, "visual_scene"):
            scene = Scene(element.get("id"), [self._parse_node(node, registry) for node in _children(element, "node")], element, self)
            self.scenes.append(scene)
        for scene in self.scenes:
            for node in scene.nodes: self._resolve_instances(node, registry)
        active = _child(root, "scene")
        instance = _child(active, "instance_visual_scene") if active is not None else None
        active_id = instance.get("url").lstrip("#") if instance is not None else None
        self.scene = next((scene for scene in self.scenes if scene.id == active_id), self.scenes[0] if self.scenes else None)
