"""Scene graph objects for visual-scene extraction."""

from __future__ import annotations

import numpy as np

from . import _lib


class GeometryNode:
    def __init__(self, geometry, materials=None):
        self.geometry = geometry
        self.materials = materials or []


class ControllerNode:
    def __init__(self, controller, materials=None, skeletons=None):
        self.controller = controller
        self.materials = materials or []
        self.skeletons = skeletons or []


class NodeNode:
    def __init__(self, node):
        self.node = node


class Node:
    def __init__(self, id=None, children=None, transforms=None, name=None, sid=None, xmlnode=None):
        self.id, self.name, self.sid = id, name or id, sid
        self.children = [] if children is None else children
        self.transforms = [] if transforms is None else transforms
        self.xmlnode = xmlnode

    @property
    def matrix(self):
        result = np.eye(4, dtype=np.float64)
        for transform in self.transforms:
            right = np.ascontiguousarray(transform, dtype=np.float64).reshape(1, 4, 4)
            left = result.reshape(1, 4, 4)
            product = np.empty((1, 4, 4), dtype=np.float64)
            _lib.lib().mpc_mat4_multiply(_lib.addr(left), _lib.addr(right), _lib.addr(product), 1)
            result = product[0]
        return result

    def iter_nodes(self, parent_matrix=None):
        parent = np.eye(4) if parent_matrix is None else parent_matrix
        world = parent @ self.matrix
        yield self, world
        for child in self.children:
            if isinstance(child, Node):
                yield from child.iter_nodes(world)
            elif isinstance(child, NodeNode):
                yield from child.node.iter_nodes(world)


class Scene:
    def __init__(self, id=None, nodes=None, xmlnode=None, collada=None):
        self.id = id
        self.nodes = nodes or []
        self.xmlnode = xmlnode
        self.collada = collada

    def iter_nodes(self):
        for node in self.nodes:
            yield from node.iter_nodes()
