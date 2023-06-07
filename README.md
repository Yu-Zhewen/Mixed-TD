# Mixed Tensor Decompostion (FPL 2023)
This is the implementation of [Mixed-TD: Efficient Neural Network Accelerator with Layer-Specific Tensor Decomposition](https://arxiv.org/abs/2306.05021).
# Introduction
Mixed-TD is a framework that maps CNNs onto FPGAs based on a novel tensor decomposition method. The proposed method applies layer-specific Singular Value Decomposition (SVD) and Canonical Polyadic Decomposition (CPD) in a mixed manner, achieving 1.73x to 10.29x throughput per DSP compared to state-of-the-art accelerators.

|                                               |                                               |
|-----------------------------------------------|-----------------------------------------------|
| <img width="200" src="figures/svd.png"> | <img width="200" src="figures/cpd.png"> | 


# Get Started

```
git submodule update --init --recursive
conda create -n mixed-td python=3.10
conda activate mixed-td 
pip install -r requirements.txt
export FPGACONVNET_OPTIMISER=~/Mixed-TD/fpgaconvnet-optimiser
export FPGACONVNET_MODEL=~/Mixed-TD/fpgaconvnet-optimiser/fpgaconvnet-model
export PYTHONPATH=$PYTHONPATH:$FPGACONVNET_OPTIMISER:$FPGACONVNET_MODEL
```

# Search Configuration
```
python search.py --gpu 0
```

# Results
todo

# Citation
```
@article{yu2023mixed,
  title={Mixed-TD: Efficient Neural Network Accelerator with Layer-Specific Tensor Decomposition},
  author={Yu, Zhewen and Bouganis, Christos-Savvas},
  journal={arXiv preprint arXiv:2306.05021},
  year={2023}
}
```
