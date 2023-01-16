# Doptrack estimate

_Add description of the purpose of the software_

## Installation

**Requirements**
- Anaconda/miniconda installation
- Git 

We recommend to use conda for managing the installation of the required dependencies. Please consult the documentation to install either [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.conda.io/en/main/miniconda.html).

Run the line below in your terminal to verify that conda is installed:
```
conda --version
```

We recommend using mamba for better performance when installing the dependencies. To install mamba in the base conda environment, execute
```
conda install -n base -c conda-forge mamba -y
```

Download the Doptrack estimate repository with the assignments and data using
```
git clone git@github.com:DopTrack/estimate.git
```

Then, move inside the `estimate` folder and install the dependencies with
```
mamba env create -f environment.yml
```

Note, the installation can take a while (~15 min) as the depedency `tudatpy` needs to download additional data files during installation. 

Finally, to access the installed dependencies, activate the `doptrack-estimate` environment with
```
conda activate doptrack-estimate
```

## Run notebooks

The installation comes with Jupyterlab. To start jupyterlab, activate the `doptrack-estimate` environment and run the following command within the repository directory

```
jupyter lab
```

Data and metadata can be found in the tar files and is extracted during the execution of the scripts.
