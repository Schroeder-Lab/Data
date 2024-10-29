# Data
For pre-processing raw data (two-photon, ephys, bonsai, ...) and loading/saving data. To start with, open the file main_preprocess.py and follow the instructions in the comments.

For pre-processing of bouton data (using a different method for registration use main_zregister.npy and follow the instructions there. This will z-register and detect ROIs. After it run use suite2p to curate the folder 'plane'.
After curation use main_preprocess.npy to finalise processing. Do not use a z-stack in this case.
