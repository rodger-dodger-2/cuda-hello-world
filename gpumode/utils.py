"""
Minimal shim that mimics the gpumode.com harness (make_match_reference,
DeterministicContext) so we can develop and benchmark locally.
"""
import torch
from typing import Callable
from task import input_t, output_t


class DeterministicContext:
    """Force deterministic CUDA ops for the reference kernel."""
    def __enter__(self):
        self._prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        return self
    def __exit__(self, *_):
        torch.use_deterministic_algorithms(self._prev, warn_only=True)


def make_match_reference(ref_fn: Callable[[input_t], output_t]):
    """
    Returns a checker that compares a candidate output to the reference.
    Mirrors the gpumode check_implementation signature.
    """
    def check(candidate_fn: Callable[[input_t], output_t],
              data: input_t,
              rtol: float = 1e-3,
              atol: float = 1e-3) -> bool:
        ref_out  = ref_fn(data)
        cand_out = candidate_fn(data)
        match = torch.allclose(
            cand_out.float(), ref_out.float(), rtol=rtol, atol=atol
        )
        if not match:
            print(f"  MISMATCH  ref={ref_out.item():.6f}  got={cand_out.item():.6f}  "
                  f"diff={abs(ref_out.item()-cand_out.item()):.2e}")
        return match
    return check
