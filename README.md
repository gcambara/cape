# CAPE 🌴
PyTorch implementation of [Continuous Augmented Positional Embeddings](https://arxiv.org/abs/2106.03143) (CAPE). Enhance your Transformers with easy-to-use augmentations for your positional embeddings! 

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
import torch.nn as nn
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
# Pos scale is 0.01 since feats are strided between each other in 0.01 s.
# Freq scale is 30 to ensure that 30 ms queries are possible with long audios
from cape import CAPE1d
pos_emb = CAPE1d(d_model=512, max_global_shift=60.0, 
                 max_local_shift=1.0, max_global_scaling=1.1, 
                 normalize=True, pos_scale=0.01,
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
@article{likhomanenko2021cape,
  title={CAPE: Encoding Relative Positions with Continuous Augmented Positional Embeddings},
  author={Likhomanenko, Tatiana and Xu, Qiantong and Collobert, Ronan and Synnaeve, Gabriel and Rogozhnikov, Alex},
  journal={arXiv preprint arXiv:2106.03143},
  year={2021}
}
```
