import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import pandas as pd
import os
import glob
import pickle
import traceback


from Data.Ephys.runners_ephys import *
from Data.Bonsai.extract_data import *
from Data.Ephys.user_defs_ephys import *
from Data.Ephys.extract_ephys import *

# %% load directories and processing ops

# Please change the values in define_directories and create_processing_ops in
# module folder_defs.
dirs = define_directories()

csvDir = dirs["dataDefFile"]
preprocessedDataDir = dirs["preprocessedDataDir"]
metadataDir = dirs["metadataDir"]
tempEphysDir = dirs["tempEphysDir"]

pops = create_ephys_processing_ops()

# %% read database

# In the file the values should be Name, Date, Experiment, ignore and save
# directory (if none default is wanted), sync (True,False), process (True,False)
database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Experiments": str, 
        "SaveDir": str,
        "Sync": bool,
        "Process": bool
    }
)


# %% run over data base
for i in range(len(database)):

    # Reads lines from database and generates directories for each line.
    # If Sync is True, sincronises arduino and nidaq streams to ephys time.
    # If Process is True, extracts Bonsai data. 
    
    (                   
        ephysDirectory,
        metadataDirectory,
        preprocessedDirectory,
        saveDirectory,
    ) = read_csv_produce_directories_ephys(
        database.loc[i], metadataDir,preprocessedDataDir
    )
    
    if database.loc[i]["Sync"]:
        
        # if False in the column "Sync", synchronisation of those experiments is
        # skipped.   
        
        try:
            print(f"Synchronising directories:\n{database.loc[i]}\n")
            
            synchronise_streams_with_ephys(database.loc[i], pops, ephysDirectory,
                                          metadataDirectory, saveDirectory, tempEphysDir)       
    
        except Exception:
            print('Error in syncronization' + str(database.loc[i]))
            print('Review directories')
            
                         
    if database.loc[i]["Process"]:
       
        # if False in the column "Process", the processing of those experiments is
        # skipped.   
        
        print("reading bonsai data")
        
        try:   
            #Find experiment folders dirs
            if database.loc[i].Experiments == 'all':
                #find experiment folders only, assuming that are all digits
                experiments  = sorted([os.path.join(metadataDirectory,x) 
                                       for x in os.listdir(metadataDirectory)
                                       if x.isdigit()],
                                       key=lambda x: int(os.path.basename(x)))
            else:    
                experiments = [os.path.join(metadataDirectory,x) 
                               for x in database.loc[i].Experiments.split(',')] 
                
                           
            ops = {
                    'Experiments':experiments,
                    }


            process_metadata_directory_ephys(
                    metadataDirectory, ops, pops, preprocessedDirectory, saveDirectory
                )
            
        except Exception:
            print("Could not process due to errors, moving to next batch.")
            print(traceback.format_exc())

    else:
        print(f"Skipping Bonsai data extraction:\n{database.loc[i]}\n")
       
