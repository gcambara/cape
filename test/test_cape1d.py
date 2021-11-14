import torch
import torch.nn as nn
from cape import CAPE1d

def test_sinusoidal_positional_encoding():
    pos_emb = CAPE1d(d_model=512)

    print("Checking expected default arguments for CAPE1d...")
    assert pos_emb.max_global_shift == 0.0, f"Error! Expected max global shift = {0.0} | Received max global shift = {pos_emb.max_global_shift}"
    assert pos_emb.local_shift == False, f"Error! Expected local shift = {False} | Received local shift = {pos_emb.local_shift}"
    assert pos_emb.max_global_scaling == 1.0, f"Error! Expected max global scaling = {1.0} | Received max global scaling = {pos_emb.max_global_scaling}"
    assert pos_emb.normalize == False, f"Error! Expected normalize = {False} | Received normalize = {pos_emb.normalize}"
    assert pos_emb.pos_scale == 1.0, f"Error! Expected position scale = {1.0} | Received position scale = {pos_emb.pos_scale}"
    assert pos_emb.freq_scale == 1.0, f"Error! Expected frequency scale = {1.0} | Received frequency scale = {pos_emb.freq_scale}"
    assert pos_emb.batch_first == False, f"Error! Expected batch first = {False} | Received batch first = {pos_emb.batch_first}"

    print("Checking correct dimensionality input/output for batch_size = False...")
    exp_shape = (10, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output for batch_size = True...")
    pos_emb = CAPE1d(d_model=512, batch_first=True)
    exp_shape = (32, 10, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

def test_cape1d():
    pos_emb = CAPE1d(d_model=512, max_global_shift=60, local_shift=True, max_global_scaling=2.1, 
                     normalize=True, pos_scale=0.01, freq_scale=30, batch_first=False)

    print("Checking correct dimensionality input/output for batch_size = False...")
    exp_shape = (10, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output for batch_size = True...")
    pos_emb = CAPE1d(d_model=512, max_global_shift=60, local_shift=True, max_global_scaling=2.1, 
                     normalize=True, pos_scale=0.01, freq_scale=30, batch_first=True)
    exp_shape = (32, 10, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

def test_augment_positions():
    print("Checking correct normalization of positions...")
    batch_size, n_tokens = 128, 200
    pos_scale, freq_scale = 1.0, 1.0
    pos_emb = CAPE1d(d_model=512, max_global_shift=0.0, local_shift=False, max_global_scaling=1.0, 
                    normalize=True, pos_scale=pos_scale, freq_scale=freq_scale, batch_first=False)

    positions = (torch.full((batch_size, 1), pos_scale) * torch.arange(n_tokens).unsqueeze(0))
    positions = pos_emb._augment_positions(positions)

    assert positions.mean() == 0.0, f"Error! After normalization expected mean = {0.0} | Received mean = {positions.mean()}"

