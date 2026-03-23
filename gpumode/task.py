from typing import Tuple
import torch

input_t = Tuple[torch.Tensor, torch.Tensor]   # (data, output_buffer)
output_t = torch.Tensor                        # scalar float32
