# -*- coding: utf-8 -*-

"""
Created on Mon Nov 20 09:35:05 2022

@author: LABadmin
"""

import pandas as pd

def define_directories():
    """
    Creates variables which contain the strings of important directory paths needed for the preprocessing.
    Note that the directory paths specified by the user can be of any shape and does not have to abide by
    the default input seen below (i.e using double backslash).

    Returns
    -------
    dataDefFile : str ["YourDirectoryPath\preprocess.csv"]
        The preprocess csv file location and name. Ths file has to contain the following information:
           - Name: animal ID.
           - Date: recording session (YYYY-MM-DD).
           - Experiments: experiment ID to process. If 'all', will process, all experiments performed in the session.
           - SaveDir: where to save the preprocessed files, if left blank will save in preprocessedDataDir.
           - Sync: whether to syncronise experiments time (arduino and nidaq times) to ephys time. (TRUE or FALSE)
           - Process: wheter to process stimulus and behavioral data for given experiments (TRUE or FALSE).

    preprocessedDataDir: str ["YourOutputDirectoryPath"]
        The default folder where the process data will be saved if 'SaveDir' in preprocess.csv file is blank.
           
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

    tempEphysDir: str ["YourLocalTemporaryDirectoryPath"] NOT IMPLEMENTED YET
        The local folder where ephys data exclusively should be downloaded from the NAS for processing.
        Data should be deleated locally afterwards. 
           
    """

    directoryDb = {
        "dataDefFile": "C:\\Software\\Data\\Ephys\\preprocess.csv", 
        "preprocessedDataDir": r'D:\Cloud\Box\Florencia\Experiments\TestPreprocessingPipeline\Output', #"D:\\Cloud\\Box\\Florencia\\Experiments\\Data", #this will be Z:\ProcessedData
        "metadataDir":  "D:\\Experiments\\" #this will be Z:\RawData
        #"tempEphysDir": "e.g. E:\Temp" Not implemented yet
    }
    return (
        directoryDb  # dataDefFile, preprocessedDataDir, zstackDir, metadataDir
    )



def create_ephys_processing_ops():
    """
        Creates the processing settings which includes:
   
        - debug: Whether or not to debug (if True, lets you see exactly at which
          lines errors occur, but parallel processing won't be done so processing
          will be slower).
        - plot: 

        Returns
        -------
        pops : dictionary [2]
            The dictionary pops which contains all the above mentioned options.


    """   
    pops = {
        "debug": True,
        "plot": True,
           }
    
    return pops