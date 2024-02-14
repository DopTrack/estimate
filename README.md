# Doptrack estimate

This code connects the tudat software via a python interface to the DopTrack data. Several assignments are constructed to explore the capabilities of the DopTrack estimate module in estimating orbits from the DopTrack data.

The purpose of the software is to estimate orbits from the DopTrack range-rate data using the Tudat Astronomical Toolbox code.

## Installation

**Requirements**
- Anaconda/miniconda installation
- Git 

We recommend to use conda for managing the installation of the required dependencies. Please consult the documentation to install either [Anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.conda.io/en/main/miniconda.html). For the SOD praktikum we advise to install Miniconda as it is less demanding for your computer.

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
git clone -b develop https://github.com/DopTrack/estimate.git
```
```diff
- NOTE FOR SOD PRACTICAL STUDENTS:
```
use the following command line instead of the above to retrieve the branch corresponding to the practical:
```
git clone -b sod_practical https://github.com/DopTrack/estimate.git
```

Then, move inside the `estimate` folder and install the dependencies with
```
mamba env create -f environment.yml
```

Note, the installation can take a while (~15 min) as the dependency `tudatpy` needs to download additional data files during installation. 

Finally, to access the installed dependencies, activate the `doptrack-estimate` environment with
```
conda activate doptrack-estimate
```

## Run notebooks

The installation comes with Jupyterlab. To start jupyterlab, activate the `doptrack-estimate` environment and run the following command within the repository directory

```
jupyter lab
```

Data and metadata can be found in the tar files and are extracted during the execution of the scripts.

## Authors 
This Software has been developed on ideas and software from the following developers/contributors:

- **Sam Fayolle**, Technische Universiteit Delft (Developer)
- **Bart Root**  ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7742-1434](https://orcid.org/0000-0001-7742-1434), Technische Universiteit Delft (Developer)
- **Martin Søndergaard**, Technische Universiteit Delft (Developer)

## License
The contents of this repository are licensed under a GNU General Public License v3.0 (see LICENSE file).

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program “DopTrack-estimate”. Doptrack-estimate is a python package to perform orbit estimation of DopTrack-derived Doppler data written by the Author(s). Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2021, B.C. Root

## References

The main part of the code is based on the software project Tudat: TU Delft Astronomical Toolbox and can be found on https://docs.tudat.space/en/latest/

Furthermore, the DopTrack/estimate code is part of a larger DopTrack codebase. You are free to acknowlege us!

## Would you like to contribute?
If you have any comments, feedback, or recommendations, feel free to reach out by sending an email to b.c.root@tudelft.nl

If you would like to contribute directly, you are welcome to fork this repository.

Thank you and enjoy!
