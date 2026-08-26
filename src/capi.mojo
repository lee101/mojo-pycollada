"""Hot-path COLLADA kernels exposed through a deliberately small C ABI."""

from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def expand_range(
    src: Ptr, indices: IPtr, dst: Ptr, start: Int, stop: Int, stride: Int
):
    for i in range(start, stop):
        var index = Int(indices.unsafe_load(i))
        var source_base = index * stride
        var destination_base = i * stride
        if stride == 3:
            dst.unsafe_store(destination_base, src.unsafe_load(source_base))
            dst.unsafe_store(destination_base + 1, src.unsafe_load(source_base + 1))
            dst.unsafe_store(destination_base + 2, src.unsafe_load(source_base + 2))
        else:
            var j = 0
            while j + W <= stride:
                dst.unsafe_store(
                    destination_base + j, src.unsafe_load[width=W](source_base + j)
                )
                j += W
            while j < stride:
                dst.unsafe_store(
                    destination_base + j, src.unsafe_load(source_base + j)
                )
                j += 1


@export("mpc_expand_f64")
def mpc_expand_f64(
    src_addr: Int, indices_addr: Int, dst_addr: Int, count: Int, stride: Int
) abi("C"):
    var src = Ptr(unsafe_from_address=src_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    expand_range(src, indices, dst, 0, count, stride)


@export("mpc_mat4_multiply")
def mpc_mat4_multiply(
    left_addr: Int, right_addr: Int, dst_addr: Int, count: Int
) abi("C"):
    var left = Ptr(unsafe_from_address=left_addr)
    var right = Ptr(unsafe_from_address=right_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for item in range(count):
        var base = item * 16
        for row in range(4):
            for col in range(4):
                var total = 0.0
                for k in range(4):
                    total += left.unsafe_load(base + row * 4 + k) * right.unsafe_load(
                        base + k * 4 + col
                    )
                dst.unsafe_store(base + row * 4 + col, total)


@export("mpc_skin_vertices")
def mpc_skin_vertices(
    positions_addr: Int,
    matrices_addr: Int,
    offsets_addr: Int,
    joints_addr: Int,
    weights_addr: Int,
    dst_addr: Int,
    vertex_count: Int,
) abi("C"):
    var positions = Ptr(unsafe_from_address=positions_addr)
    var matrices = Ptr(unsafe_from_address=matrices_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var joints = IPtr(unsafe_from_address=joints_addr)
    var weights = Ptr(unsafe_from_address=weights_addr)
    var dst = Ptr(unsafe_from_address=dst_addr)
    for vertex in range(vertex_count):
        var x = positions.unsafe_load(vertex * 3)
        var y = positions.unsafe_load(vertex * 3 + 1)
        var z = positions.unsafe_load(vertex * 3 + 2)
        var ox = 0.0
        var oy = 0.0
        var oz = 0.0
        for influence in range(
            Int(offsets.unsafe_load(vertex)), Int(offsets.unsafe_load(vertex + 1))
        ):
            var base = Int(joints.unsafe_load(influence)) * 16
            var weight = weights.unsafe_load(influence)
            ox += weight * (
                matrices.unsafe_load(base) * x
                + matrices.unsafe_load(base + 1) * y
                + matrices.unsafe_load(base + 2) * z
                + matrices.unsafe_load(base + 3)
            )
            oy += weight * (
                matrices.unsafe_load(base + 4) * x
                + matrices.unsafe_load(base + 5) * y
                + matrices.unsafe_load(base + 6) * z
                + matrices.unsafe_load(base + 7)
            )
            oz += weight * (
                matrices.unsafe_load(base + 8) * x
                + matrices.unsafe_load(base + 9) * y
                + matrices.unsafe_load(base + 10) * z
                + matrices.unsafe_load(base + 11)
            )
        dst.unsafe_store(vertex * 3, ox)
        dst.unsafe_store(vertex * 3 + 1, oy)
        dst.unsafe_store(vertex * 3 + 2, oz)
