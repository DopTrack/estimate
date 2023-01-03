Installation
---------------

First install TudatPy following the steps below.

Start by verifying that conda is installed (otherwise check installation guide in Anaconda documentation). Then download the required environment:
```
wget https://tudat-space.readthedocs.io/en/latest/_downloads/dfbbca18599275c2afb33b6393e89994/environment.yaml
```

Create and activate environment
```
conda env create -f environment.yaml
conda activate tudat-space
```

Make sure to install the development version of Tudat (most up-to-date)
```
conda install -c tudat-team/label/dev tudatpy
```

Then get the DopTrack software package:
```
git clone git@github.com:DopTrack/Process.git
```

and install it inside the newly created environment:
```
cd Process
pip install -e .
```

Go back to main folder
```
cd ..
```

Download code
---------------

Then get estimate repository:
```
git clone git@github.com:DopTrack/estimate.git
```

Before running anything, make sure to extract the contents of the compressed folders data.tar.xz and metadata.tar.xz on which the current scripts rely.

Educational course material
---------------

To run the practical you need to install additional elements:

Make sure to install the Jupyter Lab software within the environment as well
```
conda install -c conda-forge jupyterlab
```

In one of the Jupyter notebooks you need a Cartopy python package to install
```
pip3 install cartopy
```