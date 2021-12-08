# CAPE 🌴 ![pylint](https://img.shields.io/github/workflow/status/gcambara/cape/Pylint?label=pylint) ![pytest](https://img.shields.io/github/workflow/status/gcambara/cape/Pytest?label=pytest)
PyTorch implementation of [Continuous Augmented Positional Embeddings](https://arxiv.org/abs/2106.03143) (CAPE), by Likhomanenko et al. Enhance your Transformer positional embeddings with easy-to-use augmentations! 

## Setup 🔧
Install from source:
```
git clone https://github.com/gcambara/cape.git
cd cape
pip install --editable ./
```

## Usage 📖
Ready to go along with PyTorch's official implementation of [Transformers](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html). Default initialization behaves identically as sinusoidal positional embeddings.

```python
from torch import nn
from cape import CAPE1d

pos_emb = CAPE1d(d_model=512)
transformer = nn.Transformer(d_model=512)

x = torch.randn(10, 32, 512) # seq_len, batch_size, n_feats
x = pos_emb(x) # forward sums the positional embedding by default
x = transformer(x)

# Alternatively, you can get positional embeddings separately
x = torch.randn(10, 32, 512)
scale = 512**0.5
pos_emb = pos_emb.compute_pos_emb(x)
x = (scale * x) + pos_emb
x = transformer(x)
```

Let's see a few examples of CAPE initialization for different modalities, inspired by the original [paper](https://arxiv.org/abs/2106.03143) experiments.

### CAPE for text 🔤

```CAPE1d``` is ready to be applied to text. Padding is supported by indicating
the length of samples in the forward method, with the ```x_lengths``` argument.

```python
from cape import CAPE1d
pos_emb = CAPE1d(d_model=512, max_global_shift=5.0, 
                 max_local_shift=0.5, max_global_scaling=1.03, 
                 normalize=False)

# Case 1: no padding
x = torch.randn(10, 32, 512) # seq_len, batch_size, n_feats
x = pos_emb(x)

# Case 2: padding, e.g. although seq_len is 10,
# the original length of samples is 7, and they
# have been padded until 10.
# This can be specified in the forward method.
x_lengths = torch.ones(32)*7
x = pos_emb(x, x_lengths=x_lengths)
```

### CAPE for audio 🎙️
```CAPE1d``` for audio is applied similarly to audio. 
Use ```positions_delta``` argument to set the separation in seconds
between time steps, and ```x_lengths``` for indicating sample 
durations in case there is padding.

```python
# Max global shift is 60 s.
# Max local shift is set to 0.5 to maintain positional order.
# Max global scaling is 1.1, according to WSJ recipe.
# Freq scale is 30 to ensure that 30 ms queries are possible with long audios
from cape import CAPE1d
pos_emb = CAPE1d(d_model=512, max_global_shift=60.0, 
                 max_local_shift=0.5, max_global_scaling=1.1, 
                 normalize=True, freq_scale=30.0)

# Case 1: no padding & same hop size for every sample in the batch
# E.g. the feature extraction algorithm uses a stride of 30 ms
x = torch.randn(100, 32, 512) # seq_len, batch_size, n_feats
x = pos_emb(x, positions_delta=0.03)

# Case 2: padding & same hop size for every sample in the batch
# E.g. the original duration of samples if 2.5 s, although they 
# have been padded to 3.0 s. Feat extraction stride is 30 ms.
x_lengths = torch.ones(32)*2.5 # we give lengths in seconds
x = pos_emb(x, x_lengths=x_lengths, positions_delta=0.03)

# Case 3: hop size is different for every sample in the batch
# E.g. first half of samples have stride of 30 ms, and the second half
# of 50 ms.
positions_delta = torch.ones(32)*0.03
positions_delta[16:] = 0.05
x = pos_emb(x, positions_delta=positions_delta)

# Case 4 (very rare): hop size is different for every sample
# in the batch, and is not constant within some samples.
# E.g. stride of 30 ms for the first half of samples, and 50 ms
# for the second half. However, the hop size of the very first sample
# linearly increases for each timestep
from einops import repeat
positions_delta = torch.ones(32)*0.03
positions_delta[16:] = 0.05
positions_delta = repeat(positions_delta, 'b -> b new_axis', new_axis=100)
positions_delta[0, :] *= torch.arange(100)
x = pos_emb(x, positions_delta=positions_delta)
```

### CAPE for ViT 🖼️
```CAPE2d``` is used for embedding positions in image patches.
Both square and non-square patches are supported.
```python
from cape import CAPE2d
pos_emb = CAPE2d(d_model=512, max_global_shift=0.5, 
                 max_local_shift=0.5, max_global_scaling=1.4)

# Case 1: square patches
x = torch.randn(16, 16, 32, 512) # height, width, batch_size, n_feats
x = pos_emb(x)

# Case 2: non-square patches
x = torch.randn(24, 16, 32, 512) # height, width, batch_size, n_feats
x = pos_emb(x)
```

## Citation ✍️
I just did this PyTorch implementation following the [paper's](https://arxiv.org/abs/2106.03143) Python code and the [Flashlight recipe](https://github.com/flashlight/flashlight/blob/cape/cape/plugin/ctc_str3_tl_main_sinpos_trick_dp01_gl60s_nopad.cpp) in C++. All the credit goes to the original authors, please cite them if you use this for your research project:
``` bibtex
@inproceedings{likhomanenko2021cape,
title={{CAPE}: Encoding Relative Positions with Continuous Augmented Positional Embeddings},
author={Tatiana Likhomanenko and Qiantong Xu and Gabriel Synnaeve and Ronan Collobert and Alex Rogozhnikov},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=n-FqqWXnWW}
}
```

## Acknowledgments 🙏
Many thanks to the paper's authors for code reviewing and clarifying doubts about the paper and the implementation. :)