import math
from einops import rearrange, repeat
import torch
from torch import nn
from torch import Tensor

class CAPE1d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0, normalize: bool = False, pos_scale: float = 1.0,
                 freq_scale: float = 1.0, batch_first: bool = False):
        super().__init__()

        assert max_global_shift >= 0, f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert max_local_shift >= 0, f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert max_global_scaling >= 1, f"""Global scaling is {max_global_scaling},
        but should be >= 1."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.normalize = normalize
        self.pos_scale = pos_scale
        self.freq_scale = freq_scale
        self.batch_first = batch_first

        freq = freq_scale * torch.exp(-2.0 * torch.floor(torch.arange(d_model) / 2)
                                      * (math.log(1e4) / d_model))
        self.register_buffer('freq', freq)

        _sin2cos_phase_shift = torch.pi / 2.0
        cos_shifts = _sin2cos_phase_shift * (torch.arange(d_model) % 2)
        self.register_buffer('cos_shifts', cos_shifts)

        self.register_buffer('content_scale', Tensor([math.sqrt(d_model)]))

    def forward(self, x: Tensor) -> Tensor:
        return (x * self.content_scale) + self.compute_pos_emb(x)

    def compute_pos_emb(self, x: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, n_tokens, _ = x.shape # b, t, c
        else:
            n_tokens, batch_size, _ = x.shape # t, b, c

        positions = repeat(self.pos_scale*torch.arange(n_tokens),
                           't -> new_axis t', new_axis=batch_size).to(x)
        positions = self.augment_positions(positions)

        positions = rearrange(positions, 'b t -> b t 1')
        product = positions * self.freq.to(x)

        pos_emb = torch.sin(product + self.cos_shifts.to(x))

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, 'b t c -> t b c')

        return pos_emb

    def augment_positions(self, positions: Tensor):
        if self.normalize:
            positions -= torch.mean(rearrange(positions[~positions.isnan()],
                                              '(b t) -> b t',
                                              b=positions.size(0),
                                              t=positions.size(1)),
                                    axis=1, keepdim=True)

        if self.training:
            batch_size, n_tokens = positions.shape

            if self.max_global_shift:
                delta = torch.FloatTensor(batch_size, 1).uniform_(-self.max_global_shift,
                                                                  self.max_global_shift)
                delta = delta.to(positions.device)
            else:
                delta = 0

            if self.max_local_shift:
                epsilon = self.pos_scale * self.max_local_shift
                delta_local = torch.FloatTensor(batch_size, n_tokens)
                delta_local = delta_local.uniform_(-epsilon,
                                                   epsilon)
                delta_local = delta_local.to(positions.device)
            else:
                delta_local = 0

            if self.max_global_scaling > 1.0:
                log_lambdas = torch.FloatTensor(batch_size, 1)
                log_lambdas = log_lambdas.uniform_(-math.log(self.max_global_scaling),
                                                   math.log(self.max_global_scaling))
                log_lambdas = log_lambdas.to(positions.device)
            else:
                log_lambdas = torch.zeros(1).to(positions.device)

            positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])

class CAPE2d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0, batch_first: bool = False):
        super().__init__()

        assert max_global_shift >= 0, f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert max_local_shift >= 0, f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert max_global_scaling >= 1, f"""Global scaling is {max_global_scaling},
        but should be >= 1."""
        assert d_model % 2 == 0, f"""The number of channels should be even,
                                     but it is odd! # channels = {d_model}."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.batch_first = batch_first

        half_channels = d_model // 2
        rho = 10 ** torch.linspace(0, 1, half_channels) 
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))
        self.register_buffer('w_x', w_x)
        self.register_buffer('w_y', w_y)

        self.register_buffer('content_scale', Tensor([math.sqrt(d_model)]))

    def forward(self, patches: Tensor) -> Tensor:
        return (patches * self.content_scale) + self.compute_pos_emb(patches)

    def compute_pos_emb(self, patches: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, patches_x, patches_y, _ = patches.shape # b, x, y, c
        else:
            patches_x, patches_y, batch_size, _ = patches.shape # x, y, b, c

        x = torch.zeros([batch_size, patches_x, patches_y])
        y = torch.zeros([batch_size, patches_x, patches_y])
        x += torch.linspace(-1, 1, patches_x)[None, :, None]
        y += torch.linspace(-1, 1, patches_y)[None, None, :]

        x, y = self.augment_positions(x, y)

        phase = torch.pi * (self.w_x * x[:, :, :, None]
                            + self.w_y * y[:, :, :, None])
        pos_emb = torch.cat([torch.cos(phase), torch.sin(phase)], axis=-1)

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, 'b x y c -> x y b c')

        return pos_emb

    def augment_positions(self, x: Tensor, y: Tensor):
        if self.training:
            batch_size, _, _ = x.shape

            if self.max_global_shift:
                x += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(x.device)
                y += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(y.device)

            if self.max_local_shift:
                diff_x = x[0, -1, 0] - x[0, -2, 0]
                diff_y = y[0, 0, -1] - y[0, 0, -2]
                epsilon_x = diff_x*self.max_local_shift
                epsilon_y = diff_y*self.max_local_shift
                x += torch.FloatTensor(x.shape).uniform_(-epsilon_x,
                                                         epsilon_x).to(x.device)
                y += torch.FloatTensor(y.shape).uniform_(-epsilon_y,
                                                         epsilon_y).to(y.device)

            if self.max_global_scaling > 1.0:
                log_l = math.log(self.max_global_scaling)
                lambdas = (torch.exp(torch.FloatTensor(batch_size, 1, 1).uniform_(-log_l,
                                                                                  log_l))
                          ).to(x.device)
                x *= lambdas
                y *= lambdas

        return x, y

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])
