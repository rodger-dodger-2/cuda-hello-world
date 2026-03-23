"""
Our solution for gpumode leaderboard #544 — vector sum reduction.

Strategy (two-pass, fp32 data pass + fp64 final combine):
  Pass 1 — 1024 blocks grid-stride the input in fp32 (full memory bandwidth),
            each writing one fp32 partial sum to scratch.
  Pass 2 — a single block upcasts the 1024 partials to fp64, reduces them,
            and writes a fp32 scalar. The fp64 combine is on a tiny buffer
            (1024 values) so the poor T4 fp64 throughput doesn't matter.

Result: memory-bandwidth-limited on the dominant pass, with fp64 precision
preserved where the rounding error actually accumulates (the final combine).
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BLOCKS     = 1024
_BLOCK_SIZE = 1024


@triton.jit
def _reduce_pass1(
    x_ptr,
    partials_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Each block grid-strides over the input in fp32, writes one fp32 partial."""
    pid    = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK_SIZE
    acc    = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base   = pid * BLOCK_SIZE
    while base < n_elements:
        off = base + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x_ptr + off, mask=off < n_elements, other=0.0)
        base += stride
    tl.store(partials_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _reduce_pass2(
    partials_ptr,
    out_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """Single block: upcast fp32 partials to fp64, reduce, write fp32 result."""
    off  = tl.arange(0, BLOCK_SIZE)
    vals = tl.load(partials_ptr + off, mask=off < n_partials, other=0.0).to(tl.float64)
    tl.store(out_ptr, tl.sum(vals, axis=0).to(tl.float32))


def solution_kernel(data: input_t) -> output_t:
    x, out = data
    n        = x.numel()
    partials = torch.empty(_BLOCKS, device=x.device, dtype=torch.float32)

    _reduce_pass1[(_BLOCKS,)](x, partials, n, BLOCK_SIZE=_BLOCK_SIZE)
    _reduce_pass2[(1,)](partials, out, _BLOCKS,
                        BLOCK_SIZE=triton.next_power_of_2(_BLOCKS))
    return out
