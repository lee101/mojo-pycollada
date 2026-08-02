"""Geometry containers and indexed primitive extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _lib


@dataclass(frozen=True)
class Input:
    offset: int
    semantic: str
    source: str
    set: int | None = None


class InputList:
    def __init__(self):
        self.inputs: list[Input] = []

    def addInput(self, offset, semantic, source, set=None):
        self.inputs.append(Input(int(offset), semantic, source, None if set is None else int(set)))

    def getList(self):
        return [(item.offset, item.semantic, item.source, item.set) for item in self.inputs]


class Primitive:
    def __init__(self, indices, material=None, inputs=None, source_by_id=None):
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.material = material
        self.inputs = inputs or InputList()
        self._source_by_id = source_by_id or {}
        self.vertex = None
        self.vertex_index = None
        self.expanded_vertex = None
        self.normal = None
        self.normal_index = None
        self.expanded_normal = None
        self.texcoordset = ()
        self.texcoord_indexset = ()
        self.expanded_texcoordset = ()
        self._extract()

    @property
    def nindices(self):
        return len(self.indices)

    def _source(self, url):
        return self._source_by_id[url.lstrip("#")]

    def _expand(self, source, index):
        index = _lib.i64(index).reshape(-1)
        values = _lib.f64(source.data)
        if values.ndim != 2 or values.shape[1] <= 0:
            raise ValueError("source data must have shape (n, positive_stride)")
        if len(index) and (index.min() < 0 or index.max() >= len(values)):
            raise ValueError("primitive index is outside its source array")
        result = np.empty((len(index), values.shape[1]), dtype=np.float64)
        # Avoid constructing pointers for zero-length user buffers.
        if len(index):
            _lib.lib().mpc_expand_f64(_lib.addr(values), _lib.addr(index), _lib.addr(result), len(index), values.shape[1])
        return result

    def _extract(self):
        texcoords: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for item in self.inputs.inputs:
            if item.semantic == "VERTEX":
                semantic = "POSITION"
            else:
                semantic = item.semantic
            source = self._source(item.source)
            index = self.indices[:, item.offset]
            shaped_index = index.reshape((-1, 3)) if isinstance(self, TriangleSet) else index.copy()
            data = source.data
            expanded = self._expand(source, index)
            if semantic == "POSITION":
                self.vertex, self.vertex_index, self.expanded_vertex = data, shaped_index, expanded
            elif semantic == "NORMAL":
                self.normal, self.normal_index, self.expanded_normal = data, shaped_index, expanded
            elif semantic == "TEXCOORD":
                texcoords[item.set or 0] = (data, shaped_index, expanded)
        texcoord_data, texcoord_index, expanded_texcoords = [], [], []
        for set_number in sorted(texcoords):
            data, index, expanded = texcoords[set_number]
            texcoord_data.append(data)
            texcoord_index.append(index)
            expanded_texcoords.append(expanded)
        self.texcoordset = tuple(texcoord_data)
        self.texcoord_indexset = tuple(texcoord_index)
        self.expanded_texcoordset = tuple(expanded_texcoords)


class TriangleSet(Primitive):
    @property
    def ntriangles(self):
        return len(self.indices) // 3


class Polylist(Primitive):
    def __init__(self, indices, vcount, material=None, inputs=None, source_by_id=None):
        self.vcount = np.ascontiguousarray(vcount, dtype=np.int64)
        super().__init__(indices, material, inputs, source_by_id)

    @property
    def npolygons(self):
        return len(self.vcount)

    def triangleset(self):
        rows = []
        cursor = 0
        for count in self.vcount:
            for corner in range(1, int(count) - 1):
                rows.extend((self.indices[cursor], self.indices[cursor + corner], self.indices[cursor + corner + 1]))
            cursor += int(count)
        return TriangleSet(np.asarray(rows, dtype=np.int64), self.material, self.inputs, self._source_by_id)


class Geometry:
    def __init__(
        self, collada=None, id=None, name=None, sourcebyid=None, primitives=None,
        xmlnode=None, double_sided=False, *, sourceById=None,
    ):
        if id is None and isinstance(collada, str):
            collada, id = None, collada
        self.id = id
        self.name = name or id
        self.sourceById = sourcebyid if sourcebyid is not None else (sourceById or {})
        self.primitives = [] if primitives is None else primitives
        self.xmlnode = xmlnode
        self.collada = collada
        self.double_sided = double_sided

    def createTriangleSet(self, indices, inputlist, materialid=None):
        primitive = TriangleSet(indices, materialid, inputlist, self.sourceById)
        self.primitives.append(primitive)
        return primitive

    def createPolylist(self, indices, vcount, inputlist, materialid=None):
        primitive = Polylist(indices, vcount, materialid, inputlist, self.sourceById)
        self.primitives.append(primitive)
        return primitive
