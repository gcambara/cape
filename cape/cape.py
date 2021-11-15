import math
import torch
import torch.nn as nn
from torch import Tensor

class CAPE1d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0, max_global_scaling: float = 1.0, normalize: bool = False, pos_scale: float = 1.0, freq_scale: float = 1.0, batch_first: bool = False):
        super().__init__()
        assert max_global_scaling >= 1, f"Global scaling is {max_global_scaling}, but should be >= 1."

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.normalize = normalize
        self.pos_scale = pos_scale
        self.freq_scale = freq_scale
        self.batch_first = batch_first

        freq = self.freq_scale * torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
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
            positions -= torch.mean(positions[~positions.isnan()].view(positions.shape), axis=1, keepdim=True)  

        if self.training:
            batch_size, n_tokens = positions.shape
            delta = torch.FloatTensor(batch_size, 1).uniform_(-self.max_global_shift, self.max_global_shift).to(positions.device)
            if self.max_local_shift:
                delta_local = torch.FloatTensor(batch_size, n_tokens).uniform_(-(self.pos_scale*self.max_local_shift) / 2.0, (self.pos_scale*self.max_local_shift) / 2.0).to(positions.device)
            else:
                delta_local = 0
            log_lambdas = torch.FloatTensor(batch_size, 1).uniform_(-math.log(self.max_global_scaling), math.log(self.max_global_scaling)).to(positions.device)

            positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions

class CAPE2d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0, max_global_scaling: float = 1.0, batch_first: bool = False):
        super().__init__()
        assert d_model % 2 == 0, f"The number of channels should be even, but it is odd! # channels = {d_model}."

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.batch_first = batch_first

        half_channels = d_model // 2
        rho = 10 ** (torch.arange(1, half_channels + 1) / half_channels)
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))
        self.register_buffer('w_x', w_x)
        self.register_buffer('w_y', w_y)

    def forward(self, patches: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, patches_x, patches_y, n_feats = patches.shape
        else:
            patches_x, patches_y, batch_size, n_feats = patches.shape

        x = torch.zeros([batch_size, patches_x, patches_y])
        y = torch.zeros([batch_size, patches_x, patches_y])
        x += torch.linspace(-1, 1, patches_x)[None, :, None]
        y += torch.linspace(-1, 1, patches_y)[None, None, :]

        x, y = self._augment_positions(x, y)

        phase = torch.pi * (self.w_x * x[:, :, :, None] + self.w_y * y[:, :, :, None])
        pos_emb = torch.cat([torch.cos(phase), torch.sin(phase)], axis=-1)

        if not self.batch_first:
            pos_emb = pos_emb.permute(1, 2, 0, 3)

        return pos_emb

    def _augment_positions(self, x: Tensor, y: Tensor):
        if self.training:
            batch_size, _, _ = x.shape
            x += torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift, self.max_global_shift).to(x.device)
            y += torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift, self.max_global_shift).to(y.device)
            
            if self.max_local_shift:
                diff = x[0, -1, 0] - x[0, -2, 0]
                x += torch.FloatTensor(x.shape).uniform_(-(diff*self.max_local_shift) / 2.0, (diff*self.max_local_shift) / 2.0).to(x.device)
                y += torch.FloatTensor(y.shape).uniform_(-(diff*self.max_local_shift) / 2.0, (diff*self.max_local_shift) / 2.0).to(y.device)

            lambdas = torch.exp(torch.FloatTensor(batch_size, 1, 1).uniform_(-math.log(self.max_global_scaling), math.log(self.max_global_scaling))).to(x.device)
            x *= lambdas
            y *= lambdas

        return x, y