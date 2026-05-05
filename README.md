# Data

For pre-processing raw data (two-photon, ephys, bonsai, ...) and loading/saving data. To start with, open the file
main_preprocess.py and follow the instructions in the comments.

## Pre-requisites
To use suite2p, use Python 3.9 and install all packages listed in requirements.txt. You can use a virtual environment.
If suite2p doesn't install because of a problem with numpy, the following may help (first activate your virtual environment):
```bash
pip install –upgrade pip setuptools wheel
pip install –only-binary=:all: numpy==2.0.2
pip install suite2p
```

## Pre-processing of two-photon data of boutons/small structures
For pre-processing of bouton data (using a different method for registration use main_zregister.npy and follow the
instructions there. This will z-register and detect ROIs. After it run use suite2p to curate the folder 'plane'.
After curation use main_preprocess.npy to finalise processing. Do not use a z-stack in this case.
