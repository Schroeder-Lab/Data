# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:35:05 2022

@author: LABadmin
"""

# define directories # Change to a different file
def define_directories():
    """
    Creates variables which contain the strings of important directory paths needed for the preprocessing.
    Note that the directory paths specified by the user can be of any shape and does not have to abide by
    the default input seen below (i.e using double backslash).

    Returns
    -------
    csvDir : str ["YourDirectoryPath\preprocess.csv"]
        The preprocess csv file location and name. Ths file has to contain the following information:
           - Name of the Animal.
           - Date of the experiment (YYYY-MM-DD).
           - ZStack directory number (within each experiment, which folder contains the Z stack).
           - IgnorePlanes (which planes to ignore, such as the flyback plane).
           - Save directory (where to save the preprocessed files, if left blank will save in the suite2p folder).
           - Whether to process the specified experiment or not (TRUE or FALSE).
           Please also have a look at the example csv on github (named example_preprocess.csv)
           
    s2pDir : str ["YourDirectoryPath"]
        The main directory where the suite2p folders are located; The assumed folder structure for the 
        subfolders in this directory is: s2pDir\\[animal]\\[date]\\suite2p\\[folders for each plane]\\
            The folders for each plane have to contain these files:
            - Registered movie in the form of a binary file (data.bin). Make sure this is
              present as it is IMPORTANT FOR Z STACK REGISTRATION but bear in mind it is a very large file.
            - Fluorescence traces for each ROI (F.npy).
            - Neuropil traces for each ROI (Fneu.npy).
            - The iscell file which indicates whether an ROI was classified as a cell or not (iscell.npy).
            - The ops file which indicates all the suite2p input settings specified by the user (ops.npy).
            - The stat file which indicates all the suite2p metadata (such as ROI locations in XY plane) (stat.npy).
        
            
    zstackDir : str ["YourDirectoryPath"]
        The main folder where the zStack is located. IMPORTANT Z STACK REGISTRATION
        
    metadataDir : str ["YourDirectoryPath"]
        The main folder where the metadata is located. This should contain:
        - NiDaqInput*.bin : the data from the NiDaq which contains information about: 
          photodiode, frameclock, pockel and piezo feedback and the sync signal to sync with Arduino
        - niDaqChannels*.csv : csv file which contains the names of the NiDaq channels
        - ArduinoInput*.csv : the data from the Arduino which contains information about:
          rotary encoder (forward movement), rotary encoder (backward movement), camera1, camera2 and
          sync signal to sync with NiDaq
        - arduinoChannels*.csv : csv file which contains the names of the Arduino channels  
        - props*.csv : what type of experiment it is and what parameters are used. For example,
          for moving gratings these would be: Ori (orientation), SFreq (spatial frequency), TFreq (temporal frequency)
          and Contrast
          Note: These files should be generated automatically when using the Bonsai scripts within this repository.

    """

    # TODO: Make dictionary
    dataDefFile = "D:\\preprocess.csv"
    preprocessedDataDir = "D:\\Suite2Pprocessedfiles\\"
    zstackDir = "Z:\\RawData\\"
    metadataDir = "Z:\\RawData\\"

    return dataDefFile, preprocessedDataDir, zstackDir, metadataDir


def create_2p_processing_ops():
    """
        Creates the processing settings which includes:
    <<<<<<< Updated upstream
        - debug: Whether or not to debug (if True, lets you see exactly at which
          lines errors occur, but parallel processing won't be done so processing
          will be slower).
        - plot: For each sorted ROI whether to plot the uncorrected, corrected,
        normalised traces, Z location and Z profile.
        - f0_percentile: The F0 percentile which determines which percentile of
        the lowest fluorescence distribution to use.
        - f0_window: The length of the rolling window in time (s) over which to
        calculate F0.
        - zcorrect_mode: The mode of Z correction such as with the Z stack
        ("Stack").
    =======
        - debug: Whether or not to debug (if True, lets you see exactly at which lines errors occur,
          but parallel processing won't be done so processing will be slower).
        - plot: For each sorted ROI whether to plot the uncorrected, corrected, normalised traces,
          Z location and Z profile.
        - f0_percentile: The F0 percentile which determines which percentile of the lowest fluorescence distribution to use.
        - f0_window: The length of the rolling window in time (s) over which to calculate F0.
        - zcorrect_mode: The mode of Z correction such as with the Z stack ("Stack").
    >>>>>>> Stashed changes
        - remove_z_extremes: Whether or not to remove the Z extremes in the traces.
        Please note: to change preprocessing settings, change the values in pops
        in this function.
        Returns
        -------
        pops : dictionary [6]
            The dictionary pops which contains all the above mentioned options.


    """
    pops = {
        "debug": True,
        "plot": False,
        "f0_percentile": 8,
        "f0_window": 300,
        "Npil_f0_window": 60,
        "zcorrect_mode": "Stack",
        "remove_z_extremes": True,
        "process_suite2p": False,
        "process_bonsai": True,
    }
    return pops


def create_ephys_processing_ops():
    pass
