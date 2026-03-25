"""Reference kernel — verbatim from gpumode leaderboard #544."""
import torch
from utils import make_match_reference, DeterministicContext
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        data, output = data
        output = data.to(torch.float64).sum().to(torch.float32)
        return output


def generate_input(size: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.randn(size, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    scale  = (torch.rand(1, device="cuda", generator=scale_gen)  * 9.9  + 0.1).item()

    input_tensor  = (data * scale + offset).contiguous()
    output_tensor = torch.empty(1, device="cuda", dtype=torch.float32)
    return input_tensor, output_tensor


check_implementation = make_match_reference(ref_kernel)
