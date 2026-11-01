"""DiffSound's physical inverse-rendering experiments on Apple Silicon."""

import torch

UPSTREAM = "3a0be14b8bfa95dcbcd98f3c233e981c713d021b"
DEVICE = torch.device("mps")
torch.set_num_threads(4)
