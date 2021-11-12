import math
import torch
import torch.nn as nn
from torch import Tensor

class CAPE1d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, local_shift: bool = False, max_global_scaling: float = 1.0, normalize: bool = False, pos_scale: float = 1.0, freq_scale: float = 1.0, batch_first: bool = False):
        super().__init__()
        assert max_global_scaling >= 1, f"Global scaling is {max_global_scaling}, but should be >= 1."

        self.max_global_shift = max_global_shift
        self.local_shift = local_shift
        self.max_global_scaling = max_global_scaling
        self.normalize = normalize
        self.pos_scale = pos_scale
        self.batch_first = batch_first

        freq = freq_scale * torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('freq', freq)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, n_tokens, n_feats = x.shape
        else:
            n_tokens, batch_size, n_feats = x.shape

        positions = (torch.full((batch_size, 1), self.pos_scale) * torch.arange(n_tokens).unsqueeze(0)).to(x)
        positions = self._augment_positions(positions) # B, T

        positions = positions.unsqueeze(-1) # B, T, 1
        product = positions * self.freq # (B, T, 1) * (C) = (B, T, C)

        pos_emb = torch.zeros(batch_size, n_tokens, n_feats, device=x.device)
        pos_emb[:, :, 0::2] = torch.sin(product)
        pos_emb[:, :, 1::2] = torch.cos(product)

        if not self.batch_first:
            pos_emb = pos_emb.transpose(0, 1)

        return pos_emb
    
    def _augment_positions(self, positions: Tensor):
        assert self.max_global_scaling >= 1

        if self.normalize:
            positions -= torch.nanmean(positions, axis=1, keepdim=True)

        if self.training:
            batch_size, n_tokens = positions.shape
            delta = torch.FloatTensor(batch_size, 1).uniform_(-self.max_global_shift, self.max_global_shift).to(positions.device)
            if self.local_shift:
                delta_local = torch.FloatTensor(batch_size, n_tokens).uniform_(-self.pos_scale / 2.0, self.pos_scale / 2.0).to(positions.device)
            else:
                delta_local = 0
            log_lambdas = torch.FloatTensor(batch_size, 1).uniform_(-math.log(self.max_global_scaling), math.log(self.max_global_scaling)).to(positions.device)

            positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions