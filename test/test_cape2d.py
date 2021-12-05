import torch
from cape import CAPE2d

def test_cape2d():
    pos_emb = CAPE2d(d_model=512, max_global_shift=0.0, max_local_shift=0.0,
                     max_global_scaling=1.0, batch_first=False)

    print("Checking correct dimensionality input/output (16x16) for batch_size = False...")
    exp_shape = (16, 16, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output (24x16) for batch_size = False...")
    exp_shape = (24, 16, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output (16x24) for batch_size = False...")
    exp_shape = (16, 24, 32, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output (16x16) for batch_size = True...")
    pos_emb = CAPE2d(d_model=512, max_global_shift=0.0, max_local_shift=0.0,
                     max_global_scaling=1.0, batch_first=True)
    exp_shape = (32, 16, 16, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output (24x16) for batch_size = True...")
    exp_shape = (32, 24, 16, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

    print("Checking correct dimensionality input/output (16x24) for batch_size = True...")
    exp_shape = (32, 16, 24, 512)
    x = torch.randn(exp_shape)
    x = pos_emb(x)
    assert exp_shape == x.shape, f"""Error! Expected shape = {exp_shape}
                                     | Received shape = {x.shape}"""

def test_augment_positions():
    print("Checking that positions order is not altered after local shifting...")
    def check_position_order(max_local_shift, batch_size=128,
                             patches_x=24, patches_y=24, expect_disorder=False):
        if expect_disorder:
            spotted_disorder = False
        pos_emb = CAPE2d(d_model=512, max_global_shift=0.0, max_local_shift=max_local_shift,
                         max_global_scaling=1.0, batch_first=False)

        x = torch.zeros([batch_size, patches_x, patches_y])
        y = torch.zeros([batch_size, patches_x, patches_y])
        x += torch.linspace(-1, 1, patches_x)[None, :, None]
        y += torch.linspace(-1, 1, patches_y)[None, None, :]

        x, y = pos_emb.augment_positions(x, y)
        for b in range(batch_size):
            for c in range(patches_y):
                pos_x = x[b, :, c]
                for t in range(patches_x - 1):
                    if not expect_disorder:
                        assert pos_x[t] < pos_x[t + 1], f"""Error! Pos x order has been altered
                                                            after local shifting with
                                                            max value {max_local_shift}.
                                                            Pos embedding = {pos_x}.
                                                            Index t = {t}
                                                            Index t + 1 = {t + 1}."""
                    else:
                        if pos_x[t] >= pos_x[t + 1]:
                            return

        for b in range(batch_size):
            for c in range(patches_x):
                pos_y = y[b, c, :]
                for t in range(patches_y - 1):
                    if not expect_disorder:
                        assert pos_y[t] < pos_y[t + 1], f"""Error! Pos y order has been altered
                                                            after local shifting with
                                                            max value {max_local_shift}.
                                                            Pos embedding = {pos_y}.
                                                            Index t = {t}
                                                            Index t + 1 = {t + 1}."""
                    else:
                        if pos_y[t] >= pos_y[t + 1]:
                            return

        if expect_disorder:
            assert spotted_disorder, f"""Error! Expected position disorder with
                                         max local shift = {max_local_shift}.
                                         However, haven't spotted any."""

    check_position_order(max_local_shift=0.00, patches_x=24, patches_y=24)
    check_position_order(max_local_shift=0.25, patches_x=24, patches_y=24)
    check_position_order(max_local_shift=0.50, patches_x=24, patches_y=24)

    check_position_order(max_local_shift=0.00, patches_x=24, patches_y=64)
    check_position_order(max_local_shift=0.25, patches_x=24, patches_y=64)
    check_position_order(max_local_shift=0.50, patches_x=24, patches_y=64)

    check_position_order(max_local_shift=0.00, patches_x=64, patches_y=24)
    check_position_order(max_local_shift=0.25, patches_x=64, patches_y=24)
    check_position_order(max_local_shift=0.50, patches_x=64, patches_y=24)

    check_position_order(max_local_shift=0.55, batch_size=1024,
                         patches_x=24, patches_y=24, expect_disorder=True)
    check_position_order(max_local_shift=1.00, batch_size=128,
                         patches_x=24, patches_y=24, expect_disorder=True)
