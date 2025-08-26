# iono-net
## Overview:
- `model`: model training scripts, caching and outputs
- `data`: data generation scripts and preprocessing. Requires MATLAB installation
- `utils`: utility scripts, including model evaluation, plot renders, and correctness verifications

## Configuration
Some paths are hard-coded - if using in a different directory, will probably have to manually change the parent directory. Assuming 2xA100 GPU configuration, but NVCC details should be handled by whatever drivers exist on current machine. While some split tasks are run simultaneously, not much in the way of truly parallel computation beyond built-in PyTorch functionality, should be mostly hardware agnostic.

## ABOUT files
Directories contain ABOUT files with details on how to use scripts and interpret outputs, as well as major files having built-in documentation.

## Packages
`environment.txt` should contain every required package for NCSU cluster deployment. Note specific version resolution depends on your implementation of conda vs pip, but that should hit most dependencies and versions. Initial progress was built using JAX, but there were problems with this package, and the most recent working versions were PyTorch implementations. While the JAX versions are left for reference, they should not be treated as reliable/working.
