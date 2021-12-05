import torch
from cape import CAPE1d

def test_sinusoidal_positional_encoding():
    pos_emb = CAPE1d(d_model=512)

    print("Checking expected default arguments for CAPE1d...")
    assert pos_emb.max_global_shift == 0.0, f"""Error! Expected max global shift = {0.0}
                                                | Received max global shift =
                                                {pos_emb.max_global_shift}"""
    assert pos_emb.max_local_shift == 0.0, f"""Error! Expected local shift = {0.0}
                                               | Received local shift =
                                               {pos_emb.max_local_shift}"""
    assert pos_emb.max_global_scaling == 1.0, f"""Error! Expected max global scaling = {1.0}
                                                  | Received max global scaling =
                                                  {pos_emb.max_global_scaling}"""
    assert pos_emb.normalize is False, f"""Error! Expected normalize = {False}
                                           | Received normalize =
                                           {pos_emb.normalize}"""
    assert pos_emb.pos_scale == 1.0, f"""Error! Expected position scale = {1.0}
                                         | Received position scale =
                                         {pos_emb.pos_scale}"""
    assert pos_emb.freq_scale == 1.0, f"""Error! Expected frequency scale = {1.0}
                                          | Received frequency scale =
                                          {pos_emb.freq_scale}"""
    assert pos_emb.batch_first is False, f"""Error! Expected batch first = {False}
                                             | Received batch first =
                                             {pos_emb.batch_first}"""

    print("Checking correct dimensionality input/output for batch_size = False...")
    exp_shape = (10, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output for batch_size = True...")
    pos_emb = CAPE1d(d_model=512, batch_first=True)
    exp_shape = (32, 10, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

def test_cape1d():
    pos_emb = CAPE1d(d_model=512, max_global_shift=60, max_local_shift=1.0, max_global_scaling=2.1,
                     normalize=True, pos_scale=0.01, freq_scale=30, batch_first=False)

    print("Checking correct dimensionality input/output for batch_size = False...")
    exp_shape = (10, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

    print("Checking correct dimensionality input/output for batch_size = True...")
    pos_emb = CAPE1d(d_model=512, max_global_shift=60, max_local_shift=1.0, max_global_scaling=2.1,
                     normalize=True, pos_scale=0.01, freq_scale=30, batch_first=True)
    exp_shape = (32, 10, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"Error! Expected shape = {exp_shape} | Received shape = {x.shape}"

def test_augment_positions():
    print("Checking correct normalization of positions...")
    batch_size, n_tokens = 128, 200
    pos_scale, freq_scale = 1.0, 1.0
    pos_emb = CAPE1d(d_model=512, max_global_shift=0.0, max_local_shift=0.0,
                     max_global_scaling=1.0, normalize=True, pos_scale=pos_scale,
                     freq_scale=freq_scale, batch_first=False)

    positions = (torch.full((batch_size, 1), pos_scale) * torch.arange(n_tokens).unsqueeze(0))
    positions = pos_emb.augment_positions(positions)

    assert positions.mean() == 0.0, f"""Error! After normalization expected mean = {0.0}
                                        | Received mean = {positions.mean()}"""

    print("Checking that positions order is not altered after local shifting...")
    def check_position_order(max_local_shift, batch_size=128, n_tokens=200, expect_disorder=False):
        if expect_disorder:
            spotted_disorder = False
        pos_scale, freq_scale = 1.0, 1.0
        pos_emb = CAPE1d(d_model=512, max_global_shift=0.0, max_local_shift=max_local_shift,
                         max_global_scaling=1.0, normalize=True, pos_scale=pos_scale,
                         freq_scale=freq_scale, batch_first=False)

        positions = (torch.full((batch_size, 1), pos_scale) * torch.arange(n_tokens).unsqueeze(0))
        positions = pos_emb.augment_positions(positions)
        for b in range(batch_size):
            pos = positions[b, :]
            for t in range(n_tokens - 1):
                if not expect_disorder:
                    assert pos[t] < pos[t + 1], f"""Error! Position order has been altered after
                                                    local shifting with
                                                    max value {max_local_shift}.
                                                    Pos embedding = {pos}.
                                                    Index t = {t}
                                                    Index t + 1 = {t + 1}."""
                else:
                    if pos[t] >= pos[t + 1]:
                        spotted_disorder = True

        if expect_disorder:
            assert spotted_disorder, f"""Error! Expected position disorder with
                                         max local shift = {max_local_shift}.
                                         However, haven't spotted any."""

    check_position_order(max_local_shift=0.00)
    check_position_order(max_local_shift=0.25)
    check_position_order(max_local_shift=0.50)
    check_position_order(max_local_shift=0.55, batch_size=1024, expect_disorder=True)
    check_position_order(max_local_shift=1.00, batch_size=1024, expect_disorder=True)
