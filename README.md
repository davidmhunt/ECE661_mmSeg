# ECE 661 Transfer Learning Project

## Pre-Installation
The radar development sandbox is installed using Python Poetry and requires a python 3.8 environment to already be installed on the device. 

### Installing Poetry:
 
1. Check to see if Python Poetry is installed. If the below command is successful, poetry is installed move on to setting up the conda environment

```
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

### Setting Up Python Environment using Conda
1. If conda isn't already installed, follow the [Conda Install Instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to install conda
2. Once conda is installed, create a new conda environment with the correct version of python
```
conda create -n ECE661_mmseg python=3.8
```

## Installation

```
git clone https://github.com/davidmhunt/ECE661_mmSeg.git --recurse-submodules
cd ECE661_mmseg
poetry install
```