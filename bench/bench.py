"""Run only as ``pixi run bench`` so the machine-wide benchmark lock is held."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import collada  # noqa: E402
from collada import _lib  # noqa: E402


def best(fn, repeat=3):
    value = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def upstream_load(path, repeats=100):
    code = """
import collada, sys, time
path, repeats = sys.argv[1], int(sys.argv[2])
collada.Collada(path)
start = time.perf_counter()
for _ in range(repeats): collada.Collada(path)
print(time.perf_counter() - start)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code, str(path), str(repeats)], cwd="/tmp", env=env, check=True, capture_output=True, text=True)
    return float(result.stdout) / repeats


def row(name, ours, upstream):
    ratio = upstream / ours
    verdict = "faster" if ours < upstream else "slower"
    print(f"| {name} | {ours * 1e3:.2f} ms | {upstream * 1e3:.2f} ms | {ratio:.2f}x {verdict} |")


def main():
    rng = np.random.default_rng(7)
    source = np.ascontiguousarray(rng.normal(size=(250_000, 3)))
    indices = np.ascontiguousarray(rng.integers(0, len(source), size=1_500_000, dtype=np.int64))
    destination = np.empty((len(indices), 3), dtype=np.float64)
    _lib.lib().mpc_expand_f64(_lib.addr(source), _lib.addr(indices), _lib.addr(destination), len(indices), 3)
    mojo_expand = best(lambda: _lib.lib().mpc_expand_f64(_lib.addr(source), _lib.addr(indices), _lib.addr(destination), len(indices), 3))
    numpy_expand = best(lambda: np.take(source, indices, axis=0, out=destination))

    fixture = ROOT / "tests" / "data" / "fixture.dae"
    collada.Collada(fixture)
    mojo_load = best(lambda: [collada.Collada(fixture) for _ in range(100)]) / 100
    pycollada_load = upstream_load(fixture)

    print("| case | mojo-pycollada | pycollada / NumPy | result |")
    print("| --- | ---: | ---: | --- |")
    row("indexed position expansion (1.5M vertices)", mojo_expand, numpy_expand)
    row("DAE geometry/controller/scene load (fixture)", mojo_load, pycollada_load)


if __name__ == "__main__":
    main()
