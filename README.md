# Doptrack estimate

## Installation

Start by verifying that conda is installed (otherwise check installation guide in Anaconda documentation):

Run the line below in your terminal to verify a conda version is installed:
```
conda --version
```

We recommend using mamba for better performance when installing the dependencies. To install mamba in the base conda environment, execute
```
conda install -n base -c conda-forge mamba -y
```

Download the repository with the assignments and data using
```
git clone git@github.com:DopTrack/estimate.git
```

Then, move inside the `estimate` folder and install the dependencies with
```
mamba env create -f environment.yaml
```

Note, the installation can take a while a the depedency `tudatpy` needs to download additional data files.

Finally, to access the installed dependencies, activate the environment with
```
conda activate doptrack-estimate
```

## Run notebooks

The installation comes with Jupyterlab. To start jupyterlab, run the following command within the activated `doptrack-estimate` environment

```
jupyter lab
```

Before running anything, make sure to extract the contents of the compressed folders data.tar.xz and metadata.tar.xz on which the current scripts rely.