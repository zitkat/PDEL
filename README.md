# PDEL
Repository for reproducibility Challenge 2020

Jiang, C., Kashinath, K., Prabhat, & Marcus, P. (2020). Enforcing physical constraints in neural networks through diffenrentiable PDE layer. ICLR 2020 Workshop DeepDiffEq.
[OpenReview](https://openreview.net/forum?id=yj3zuZa7tqM&referrer=%5BML%20Reproducibility%20Challenge%202020%5D(%2Fgroup%3Fid%3DML_Reproducibility_Challenge%2F2020))

The code base is mostly adapted from https://github.com/NVlabs/SPADE.

## Installation

The model requires the [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch), it is included as git submodule in folder `model/networks/sync_batchnorm/` use 
```bash
git clone --recurse-submodules https://github.com/zitkat/PDEL.git
```
to painlessly get it cloned along with the codebase itself. This code requires PyTorch>=1.7 and Python 3+, additionally data downloader for turbulence dataset requires [pyJHTDB](https://pypi.org/project/pyJHTDB/). You can install dependencies by
```bash
pip install -r requirements.txt
```
