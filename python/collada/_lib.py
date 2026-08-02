"""ctypes bridge for the allocation-free Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOCOLLADA_LIB") or os.path.join(ROOT, "dist", "libmojo-pycollada.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mpc_expand_f64": ([I, I, I, I, I], None),
    "mpc_mat4_multiply": ([I, I, I, I], None),
    "mpc_skin_vertices": ([I, I, I, I, I, I, I], None),
}
_loaded = None


def _mojo() -> list[str]:
    override = os.environ.get("MOJOCOLLADA_MOJO")
    if override:
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    if pixi := (shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise RuntimeError("mojo not found; set MOJOCOLLADA_MOJO")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    result = subprocess.run(
        _mojo() + ["build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode or not os.path.exists(LIB):
        raise RuntimeError((result.stderr or result.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def f64(value) -> np.ndarray:
    """Return a contiguous float64 array without lossy implicit casts."""
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"expected a safely convertible real floating dtype, got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(value) -> np.ndarray:
    """Return a contiguous int64 array without truncating indices."""
    array = np.asarray(value)
    if not np.can_cast(array.dtype, np.int64, casting="safe"):
        raise TypeError(f"expected a safely convertible integer dtype, got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be contiguous NumPy arrays")
    if not value.ctypes.data:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return int(value.ctypes.data)
