"""
Our solution for gpumode leaderboard #544 — vector sum reduction.

Strategy (two-pass, no atomics):
  Pass 1 — each Triton block grid-strides the input, accumulates in fp64,
            writes one fp64 partial sum per block to a scratch buffer.
  Pass 2 — a single block reduces the scratch buffer into the scalar output.

  No global atomics → blocks run fully in parallel; memory traffic is one
  read of the input (float32) plus a tiny scratch buffer round-trip.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# ── Pass 1: parallel partial sums ────────────────────────────────────────────
@triton.jit
def _reduce_pass1(
    x_ptr,
    partials_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0) * BLOCK_SIZE

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    base = pid * BLOCK_SIZE
    while base < n_elements:
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
        acc += x
        base += grid_stride

    partial = tl.sum(acc, axis=0)
    tl.store(partials_ptr + pid, partial)


# ── Pass 2: reduce the scratch buffer in one block ────────────────────────────
@triton.jit
def _reduce_pass2(
    partials_ptr,
    out_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_partials
    x = tl.load(partials_ptr + offsets, mask=mask, other=0.0)  # already fp64
    result = tl.sum(x, axis=0)
    tl.store(out_ptr, result.to(tl.float32))


_BLOCKS = 1024   # fixed grid — each block strides across the whole array


def solution_kernel(data: input_t) -> output_t:
    x, out = data
    n = x.numel()

    BLOCK_SIZE = 2048   # elements per tile (controls ILP inside each block)
    partials = torch.empty(_BLOCKS, device=x.device, dtype=torch.float64)

    _reduce_pass1[(1024,)](x, partials, n, BLOCK_SIZE=BLOCK_SIZE)

    # Pass-2 block size must be a power-of-2 >= _BLOCKS
    P2_BLOCK = triton.next_power_of_2(_BLOCKS)
    _reduce_pass2[(1,)](partials, out, _BLOCKS, BLOCK_SIZE=P2_BLOCK)

    return out
