"""Skin-controller extraction and Mojo linear-blend skinning."""

from __future__ import annotations

import numpy as np

from . import _lib


class Skin:
    def __init__(
        self, id, source, joint_source, inverse_bind_matrices, weights,
        vertex_weight_counts, joint_indices, weight_indices, bind_shape_matrix=None,
        geometry=None, joint_matrix_source=None, weight_source=None, vertex_weight_index=None, sourcebyid=None,
    ):
        self.id = id
        self.geometry = geometry
        self.sourcebyid = sourcebyid or {}
        self.joint_source = joint_source.id if hasattr(joint_source, "id") else joint_source
        self.joint_matrix_source = joint_matrix_source.id if hasattr(joint_matrix_source, "id") else joint_matrix_source
        self.weight_source = weight_source.id if hasattr(weight_source, "id") else weight_source
        self.weight_joint_source = self.joint_source
        self.joint_names = list(self.sourcebyid[self.joint_source].data) if self.sourcebyid else []
        self.inverse_bind_matrices = np.ascontiguousarray(inverse_bind_matrices, dtype=np.float64)
        self.weights = weight_source
        self._weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        self.vertex_weight_counts = np.ascontiguousarray(vertex_weight_counts, dtype=np.int64)
        self.vcounts = self.vertex_weight_counts
        self.joint_indices = np.ascontiguousarray(joint_indices, dtype=np.int64)
        self.weight_indices = np.ascontiguousarray(weight_indices, dtype=np.int64)
        if np.any(self.vertex_weight_counts < 0):
            raise ValueError("vertex influence counts must be non-negative")
        influence_count = int(self.vertex_weight_counts.sum())
        if len(self.joint_indices) != influence_count or len(self.weight_indices) != influence_count:
            raise ValueError("skin influence tables do not match vertex influence counts")
        if np.any(self.joint_indices < 0) or np.any(self.joint_indices >= len(self.joint_names)):
            raise ValueError("skin joint index is outside the joint table")
        if np.any(self.weight_indices < 0) or np.any(self.weight_indices >= len(self._weights)):
            raise ValueError("skin weight index is outside the weight table")
        self.vertex_weight_index = np.ascontiguousarray(
            vertex_weight_index if vertex_weight_index is not None else np.column_stack((joint_indices, weight_indices)).reshape(-1),
            dtype=np.int64,
        )
        self.bind_shape_matrix = np.eye(4) if bind_shape_matrix is None else np.asarray(bind_shape_matrix, dtype=np.float64)
        self._offsets = np.r_[0, np.cumsum(self.vertex_weight_counts, dtype=np.int64)]

    def apply_skinning(self, vertices, joint_matrices):
        vertices = _lib.f64(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        matrices = _lib.f64(joint_matrices)
        if matrices.ndim < 2 or matrices.size % 16:
            raise ValueError("joint_matrices must contain complete 4x4 matrices")
        matrices = matrices.reshape((-1, 4, 4))
        if len(matrices) < len(self.joint_names):
            raise ValueError("one joint matrix is required for every joint")
        if len(vertices) != len(self.vertex_weight_counts):
            raise ValueError("vertices must match the controller vertex count")
        packed_weights = np.ascontiguousarray(self._weights[self.weight_indices], dtype=np.float64)
        result = np.empty_like(vertices)
        if len(vertices):
            _lib.lib().mpc_skin_vertices(
                _lib.addr(vertices), _lib.addr(matrices), _lib.addr(self._offsets),
                _lib.addr(self.joint_indices), _lib.addr(packed_weights), _lib.addr(result), len(vertices),
            )
        return result

    skin = apply_skinning
