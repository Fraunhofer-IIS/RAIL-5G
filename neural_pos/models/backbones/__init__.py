import torch

from .mlp import MLP


Backbones: dict[str, torch.nn.Module] = {
    "MLP": MLP,
}