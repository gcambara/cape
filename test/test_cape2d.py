import torch
import torch.nn as nn
from cape import CAPE2d

def test_cape2d():
    pos_emb = CAPE2d(d_model=512, max_global_shift=0.0, max_local_shift=0.0, max_global_scaling=1.0, 
                     batch_first=False)

    print("Checking correct dimensionality input/output (16x16) for batch_size = False...")
    exp_shape = (16, 16, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output (24x16) for batch_size = False...")
    exp_shape = (24, 16, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output (16x24) for batch_size = False...")
    exp_shape = (16, 24, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output (16x16) for batch_size = True...")
    pos_emb = CAPE2d(d_model=512, max_global_shift=0.0, max_local_shift=0.0, max_global_scaling=1.0, 
                    batch_first=True)
    exp_shape = (32, 16, 16, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output (24x16) for batch_size = True...")
    exp_shape = (32, 24, 16, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output (16x24) for batch_size = True...")
    exp_shape = (32, 16, 24, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"