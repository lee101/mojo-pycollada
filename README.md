# mojo-pycollada

`mojo-pycollada` is a standalone, focused port of the useful extraction path in
[pycollada](https://github.com/pycollada/pycollada). It preserves the
`collada` Python import for the covered subset, reads COLLADA DAE files, and
uses Mojo for the dense work that follows parsing.

## Covered subset

`Collada(filename, ignore=None, aux_file_loader=None, zip_filename=None,
validate_output=False)` reads:

| area | coverage |
| --- | --- |
| asset | unit meter scale and up axis |
| geometry | `FloatSource`, `NameSource`, `<vertices>`, indexed `triangles`, `polylist`, position/normal/UV attributes, and `Polylist.triangleset()` |
| controller | `<skin>`, bind-shape and inverse-bind matrices, joints, weights, vertex influence tables, plus `Skin.apply_skinning()` |
| scene | visual scenes, nested nodes, translate/rotate/scale/matrix transforms, geometry/controller instances, and `instance_node` references |

The public containers retain pycollada's common names: `Geometry`,
`InputList`, `TriangleSet`, `Polylist`, `Skin`, `Scene`, `Node`,
`GeometryNode`, and `ControllerNode`. Primitive attributes such as `vertex`,
`vertex_index`, `normal`, `normal_index`, `texcoordset`, and
`texcoord_indexset` follow pycollada's source-plus-index representation.
`expanded_vertex`, `expanded_normal`, and `expanded_texcoordset` are the
contiguous, Mojo-expanded forms for rendering or compute code.

Not included: material/effect/image loading, lines/polygons beyond polylist,
morph controllers, animation, cameras/lights, writer support, XML validation,
and pycollada's bound-object/rendering helpers. Those are deliberately not
silently stubbed.

## Install and use

```bash
pixi install
pixi run build
pixi run test
```

`PYTHONPATH=python` is activated by Pixi, so this example runs from the repo:

```python
import collada
import numpy as np

mesh = collada.Collada("tests/data/fixture.dae")
triangle = mesh.geometries[0].primitives[0]
print(triangle.vertex[triangle.vertex_index])

skin = mesh.controllers[0]
joint_matrices = np.repeat(np.eye(4)[None], len(skin.joint_names), axis=0)
deformed = skin.apply_skinning(mesh.geometries[0].sourceById["pos"].data, joint_matrices)
```

## How it works

The DAE XML and Python object graph stay in Python, where COLLADA's optional
and reference-heavy structure is most naturally handled. The one Mojo
compilation unit in `src/capi.mojo` exports an ordinary C ABI. NumPy buffers
cross `ctypes` as `Int` addresses; the Mojo wrapper rebuilds mutable typed
pointers without allocating. Geometry source arrays are contiguous float64,
indices are contiguous int64, and the caller owns all output buffers.

The three kernels are indexed attribute expansion, row-major 4x4 matrix
multiplication, and packed linear-blend skinning. The latter uses a CSR-style
offset table (`vertex_weight_counts` prefix summed once) so it handles a
different influence count for every vertex without per-vertex allocation.

## Parity and performance

`pycollada 0.9.3` is installed through Pixi and the tests launch it in an
isolated interpreter, avoiding this project's `collada` package shadowing it.
They assert parity for source data and index tables, polylist triangulation,
asset data, skin influence tables, and scene transforms. Kernel results are
also checked against NumPy references.

Measured with `pixi run bench` on Linux 6.8, dual Intel Xeon E5-2697 v4
(72 logical CPUs), using Mojo `1.0.0b3.dev2026072406`, NumPy from the Pixi
environment, and pycollada 0.9.3:

| case | mojo-pycollada | pycollada / NumPy | result |
| --- | ---: | ---: | --- |
| indexed position expansion (1.5M vertices) | 13.92 ms | 82.11 ms | 5.90x faster |
| DAE geometry/controller/scene load (fixture) | 0.85 ms | 0.97 ms | 1.15x faster |

The native expansion kernel uses SIMD for contiguous source rows and chunks
large independent expansions across CPU workers. No GPU path is provided:
expansion is gather and memory-bandwidth bound, while the matrix and skinning
kernels are below the arithmetic intensity needed to outweigh device transfers.

Reproduce exactly with:

```bash
pixi run bench
```

## Development

The repository intentionally has one Mojo source file because shared-library
build cost dominates at this scale. `build/build.sh` emits
`dist/libmojo-pycollada.so`; the Python wrapper rebuilds it automatically when
the source is newer. All project commands go through Pixi:

```bash
pixi run build && pixi run test && pixi run bench
```

## License

MIT
