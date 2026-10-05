import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer, data_shape, dropout=0.0, single_antenna=False, out_dim=2, norm=True):
        super().__init__()
        self.flatten = not single_antenna
        layer.insert(0, math.prod(data_shape))
        layer.append(out_dim)

        blocks = []
        for i in range(len(layer) - 1):
            in_s, out_s = layer[i], layer[i + 1]
            parts = [nn.Linear(in_s, out_s)]
            if norm and i == 1 and len(layer) > 2:
                parts.append(nn.LayerNorm(out_s))
            if i < len(layer) - 2:
                parts.append(nn.ReLU())
            if i == 1 and dropout > 0 and i != len(layer) - 2:
                parts.append(nn.Dropout(p=dropout))
            blocks.append(nn.Sequential(*parts))
        self.net = nn.Sequential(*blocks)

    def forward(self, x, valid_mask=None):
        if valid_mask is not None:
            # For MLPs we set masked values to zero.
            x = x * valid_mask[:, None, :, None]
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        else:
            # We operate on single antennas, so we will flatten channel and taps, but keep batch and antenna dimensions.
            B, C, W, H = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B, W, C * H)
        logits = self.net(x)
        return logits