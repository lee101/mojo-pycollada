"""Behavioural checks against pycollada 0.9.x and numerical kernel references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import collada
from collada import _lib

FIXTURE = Path(__file__).parent / "data" / "fixture.dae"


def upstream_summary():
    code = '''
import collada, json, sys, numpy as np
m = collada.Collada(sys.argv[1])
g = m.geometries[0]
t = g.primitives[0]
p = g.primitives[1].triangleset()
s = m.controllers[0]
print(json.dumps({
    "asset": [m.assetInfo.unitmeter, m.assetInfo.upaxis],
    "geometry": [g.id, g.name, t.vertex.tolist(), t.vertex_index.tolist(), t.normal.tolist(), t.normal_index.tolist(), t.texcoordset[0].tolist(), t.texcoord_indexset[0].tolist(), p.vertex.tolist(), p.vertex_index.tolist()],
    "skin": [np.asarray(s.sourcebyid[s.joint_source].data).reshape(-1).tolist(), s.vcounts.tolist(), s.vertex_weight_index.tolist()],
    "scene": [m.scene.id, m.scene.nodes[0].matrix.tolist(), m.scene.nodes[0].children[0].matrix.tolist()],
}))
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(FIXTURE)], cwd="/tmp", env=env,
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_geometry_and_scene_match_pycollada():
    ours, upstream = collada.Collada(FIXTURE), upstream_summary()
    triangle, poly = ours.geometries[0].primitives
    assert [ours.assetInfo.unitmeter, ours.assetInfo.upaxis] == upstream["asset"]
    assert [ours.geometries[0].id, ours.geometries[0].name] == upstream["geometry"][:2]
    assert np.allclose(triangle.vertex, upstream["geometry"][2])
    assert np.array_equal(triangle.vertex_index, upstream["geometry"][3])
    assert np.allclose(triangle.normal, upstream["geometry"][4])
    assert np.array_equal(triangle.normal_index, upstream["geometry"][5])
    assert np.allclose(triangle.texcoordset[0], upstream["geometry"][6])
    assert np.array_equal(triangle.texcoord_indexset[0], upstream["geometry"][7])
    assert np.allclose(poly.triangleset().vertex, upstream["geometry"][8])
    assert np.array_equal(poly.triangleset().vertex_index, upstream["geometry"][9])
    assert ours.scene.id == upstream["scene"][0]
    assert np.allclose(ours.scene.nodes[0].matrix, upstream["scene"][1])
    assert np.allclose(ours.scene.nodes[0].children[0].matrix, upstream["scene"][2], atol=1e-6)


def test_controller_tables_match_pycollada():
    ours, upstream = collada.Collada(FIXTURE).controllers[0], upstream_summary()["skin"]
    assert ours.joint_names == upstream[0]
    assert np.array_equal(ours.vcounts, upstream[1])
    assert np.array_equal(ours.vertex_weight_index, upstream[2])


def test_indexed_expansion_matches_numpy_with_simd_tail():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(97, 67))
    indices = rng.integers(0, len(source), size=1003, dtype=np.int64)
    actual = np.empty((len(indices), 67))
    _lib.lib().mpc_expand_f64(_lib.addr(source), _lib.addr(indices), _lib.addr(actual), len(indices), 67)
    assert np.array_equal(actual, source[indices])


def test_indexed_expansion_matches_numpy_at_parallel_threshold():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(257, 3))
    indices = rng.integers(0, len(source), size=131_073, dtype=np.int64)
    actual = np.empty((len(indices), 3))
    _lib.lib().mpc_expand_f64(_lib.addr(source), _lib.addr(indices), _lib.addr(actual), len(indices), 3)
    assert np.array_equal(actual, source[indices])


def test_expansion_rejects_out_of_range_indices_before_entering_mojo():
    source = collada.FloatSource("positions", np.zeros((2, 3)))
    inputs = collada.InputList()
    inputs.addInput(0, "VERTEX", "#positions")
    with np.testing.assert_raises_regex(ValueError, "outside"):
        collada.TriangleSet(np.array([[0], [1], [2]]), inputs=inputs, source_by_id={"positions": source})


def test_skinning_rejects_lossy_input_and_empty_expansion_is_safe():
    skin = collada.Collada(FIXTURE).controllers[0]
    with np.testing.assert_raises(TypeError):
        skin.apply_skinning(np.zeros((4, 3), dtype=np.int64), np.repeat(np.eye(4)[None], 2, axis=0))
    source = collada.FloatSource("positions", np.empty((0, 3)))
    inputs = collada.InputList()
    inputs.addInput(0, "VERTEX", "#positions")
    primitive = collada.TriangleSet(np.empty((0, 1), dtype=np.int64), inputs=inputs, source_by_id={"positions": source})
    assert primitive.expanded_vertex.shape == (0, 3)


def test_skinning_matches_numpy_linear_blend_reference():
    mesh = collada.Collada(FIXTURE)
    skin = mesh.controllers[0]
    vertices = mesh.geometries[0].sourceById["pos"].data
    matrices = np.array([np.eye(4), [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=float)
    expected = np.empty_like(vertices)
    for vertex, point in enumerate(vertices):
        value = np.zeros(3)
        start, stop = skin._offsets[vertex:vertex + 2]
        for influence in range(start, stop):
            value += skin._weights[skin.weight_indices[influence]] * (matrices[skin.joint_indices[influence]] @ np.r_[point, 1.0])[:3]
        expected[vertex] = value
    assert np.allclose(skin.apply_skinning(vertices, matrices), expected)


def test_node_world_matrices_and_instance_references():
    mesh = collada.Collada(FIXTURE)
    walked = dict(mesh.scene.iter_nodes())
    assert set(node.id for node in walked) == {"root", "child"}
    assert np.allclose(walked[mesh.scene.nodes[0]], [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def test_scene_parser_handles_matrix_scale_and_instance_node():
    document = b"""<COLLADA><library_visual_scenes><visual_scene id='scene'>
    <node id='template'><scale>2 3 4</scale></node>
    <node id='root'><matrix>1 0 0 0 0 1 0 0 0 0 1 0 5 0 0 1</matrix><instance_node url='#template'/></node>
    </visual_scene></library_visual_scenes><scene><instance_visual_scene url='#scene'/></scene></COLLADA>"""
    scene = collada.Collada(document).scene
    root, template = scene.nodes[1], scene.nodes[0]
    assert np.allclose(root.matrix[:3, 3], [5, 0, 0])
    assert np.allclose(template.matrix, np.diag([2, 3, 4, 1]))
    assert isinstance(root.children[0], collada.NodeNode)


def test_source_accessor_offset_and_unsupported_primitives_are_explicit():
    document = b"""<COLLADA><library_geometries><geometry id='g'><mesh>
    <source id='p'><float_array>9 1 2 3</float_array><technique_common><accessor count='1' stride='3' offset='1'/></technique_common></source>
    <vertices id='v'><input semantic='POSITION' source='#p'/></vertices>
    </mesh></geometry></library_geometries></COLLADA>"""
    assert np.array_equal(collada.Collada(document).geometries[0].sourceById["p"].data, [[1, 2, 3]])
    with np.testing.assert_raises_regex(collada.ColladaError, "unsupported mesh primitive"):
        collada.Collada(b"<COLLADA><library_geometries><geometry><mesh><lines/></mesh></geometry></library_geometries></COLLADA>")
