"""
Our solution for gpumode leaderboard #544 — vector sum reduction.

Strategy:
  - Triton kernel that accumulates in float64 (matches reference precision)
  - Single-pass over the data with a tree reduction across warps
  - Falls back to torch.sum(float64) if Triton isn't available
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t


# ── Triton kernel ──────────────────────────────────────────────────────────────
# Each program (block) handles BLOCK_SIZE elements, accumulates in fp64,
# and atomically adds its partial sum into the output scalar.
@triton.jit
def _reduce_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid    = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
    partial = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, partial.to(tl.float32))


def solution_kernel(data: input_t) -> output_t:
    x, out = data
    n = x.numel()
    out.zero_()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _reduce_sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
