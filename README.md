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
x += pos_emb(x)
x = transformer(x)
```

Let's see a few examples of CAPE initialization for different modalities, inspired by the original [paper](https://arxiv.org/abs/2106.03143) experiments.

### CAPE for text 🔤
```python
from cape import CAPE1d
pos_emb = CAPE1d(d_model=512, max_global_shift=5.0, 
                 max_local_shift=1.0, max_global_scaling=1.03, 
                 normalize=False)
```

### CAPE for audio 🎙️
```python
# Max global shift is 60 s.
# Max local shift is set to 1.0 to maintain positional order.
# Max global scaling is 1.1, according to WSJ recipe.
# Pos scale is 0.03 since feats are strided between each other in 0.03 s.
# Freq scale is 30 to ensure that 30 ms queries are possible with long audios
from cape import CAPE1d
pos_emb = CAPE1d(d_model=512, max_global_shift=60.0, 
                 max_local_shift=1.0, max_global_scaling=1.1, 
                 normalize=True, pos_scale=0.03,
                 freq_scale=30.0)
```

### CAPE for ViT 🖼️
```python
from cape import CAPE2d
pos_emb = CAPE2d(d_model=512, max_global_shift=0.5, 
                 max_local_shift=1.0, max_global_scaling=1.4)
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
Many thanks to Tatiana Likhomanenko for code reviewing and clarifying doubts about the paper and the implementation. :)