"""Source arrays compatible with the useful part of :mod:`pycollada.source`."""

from __future__ import annotations

import numpy as np


class FloatSource:
    def __init__(self, id, data, components=(), xmlnode=None):
        self.id = id
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        if self.data.ndim == 1:
            self.data = self.data.reshape((-1, 1))
        self.components = tuple(components)
        self.xmlnode = xmlnode


class NameSource:
    def __init__(self, id, data, components=(), xmlnode=None):
        self.id = id
        self.data = np.asarray(data, dtype=object)
        self.components = tuple(components)
        self.xmlnode = xmlnode
