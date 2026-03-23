# Installation Guide

## 1. Create a conda virtual environment
```bash
# create conda environment
conda create -n rsgen python=3.9 -y

# activate the newly created environment
conda activate rsgen
```
## 2. Install dependencies
```bash
# install the running environment dependencies
pip install -r requirement.txt
cd diffusers
pip install -e .
