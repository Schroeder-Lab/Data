# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:35:05 2022

@author: LABadmin
"""


def define_directories():
    """
    This function creates variables which contain the file directories needed for the preprocessing.

    Returns
    -------
    csvDir : str ["Drive:\\DirectoryName\\preprocess.csv"]
        The preprocess csv file location and name. Ths file has to contain the following information:
           - Name of the Animal.
           - Date of the experiment.
           - ZStack directory number (within each experiment, which folder contains the Z stack).
           - IgnorePlanes (which planes to ignore, such as the flyback plane).
           - Save directory (where to save the preprocessed files, if left blank will save in the suite2p folder).
           - Whether to process the specified experiment or not (TRUE or FALSE).         
           
    s2pDir : str ["Drive:\\DirectoryName\\"]
        The directory where the suite2p folders are located; have to contain these files for each plane:
            - Registered movie in the form of a binary file (data.bin). Make sure this is
              present as it is IMPORTANT FOR Z STACK REGISTRATION but bear in mind it is a very large file.
            - Fluorescence traces for each ROI (F.npy).
            - Neuropil traces for each ROI (Fneu.npy).
            - The iscell file which indicates whether an ROI was classified as a cell or not (iscell.npy).
            - The ops file which indicates all the suite2p input settings specified by the user (ops.npy).
            - The stat file which indicates all the suite2p metadata (such as ROI locations in XY plane) (stat.npy).
            
    zstackDir : str ["Drive:\\DirectoryName\\]
        The main folder where the zStack is located.
    metadataDir : str ["Drive:\\DirectoryName\\]
        The main folder where the metadata is located.

    """
    csvDir = "D:\\preprocess.csv"
    s2pDir = "Z:\\Suite2Pprocessedfiles\\"
    zstackDir = "Z:\\RawData\\"
    metadataDir = "Z:\\RawData\\"

    return csvDir, s2pDir, zstackDir, metadataDir


def create_processing_ops():
    """
    Creating the processing settings which includes:
    - debug: Whether or not to debug (if True, lets you see exactly at which lines errors occur, 
      but parallel processing won't be done so processing will be slower).
    - plot: For each sorted ROI whether to plot the uncorrected, corrected, normalised traces, 
      Z location and Z profile.
    - f0_percentile: The F0 percentile which determines which percentile of the lowest fluorescence distribution to use.
    - f0_window: The length of the rolling window in time (s) over which to calculate F0.
    - zcorrect_mode: The mode of Z correction such as with the Z stack ("Stack").
    - remove_z_extremes: Whether or not to remove the Z extremes in the traces.
    Returns
    -------
    pops : dictionary [6]
        The dictionary pops which contains all the above mentioned options. 
        

    """
    pops = {
        "debug": True,
        "plot": True,
        "f0_percentile": 8,
        "f0_window": 60,
        "zcorrect_mode": "Stack",
        "remove_z_extremes": True,
    }
    return pops
