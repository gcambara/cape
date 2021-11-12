# CAPE 🌴
[Continuous Augmented Positional Embeddings](https://arxiv.org/abs/2106.03143) (CAPE) implementation for PyTorch. Enhance your Transformers with easy-to-use augmentations for your positional embeddings! 

## Setup 🔧
Requirements:
* torch >= 1.10.0

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
