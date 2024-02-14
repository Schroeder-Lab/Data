# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:49:09 2024

@author: Florencia

Idea: run cell by cell and perform corrections accordingly. Finally concat 
"""

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

# %% Load data set with issues to explore in detail

# Please change the values in define_directories and create_processing_ops in
# module folder_defs.
dirs = define_directories()

csvDir = dirs["tocorrectDefFile"]
preprocessedDataDir = dirs["preprocessedDataDir"]
metadataDir = dirs["metadataDir"]

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
        "Sync": str,
        "Process": str,
    }
)


# %% run over data base

 #Generates directories for data set to explore   
(ephysDirectory,
    metadataDirectory,
    saveDirectory,
) = read_csv_produce_directories_ephys(
    database.loc[0], metadataDir,preprocessedDataDir 
)



#%% Extract stimulus information

# 1) Load nidaq data 

di = os.path.join(metadataDirectory, database.Experiments.loc[0])

try:
    nidaqData, nidaqChans, nt = get_nidaq_channels(di, plot=pops["plot"])

    # make sure everything is in small letters
    nidaqChans = [s.lower() for s in nidaqChans]
except:
    print(traceback.format_exc())  
    print('[Process] Failed nidaq extraction. Assuming number of channels is 2')
    nidaqData, nidaqChans, nt = get_nidaq_channels(di,2, plot=pops["plot"])
    nidaqChans = ['photodiode', 'sync']
   
#Load nidaq alignment and correct times
try:
    alignmentNidaq = np.load(os.path.join(saveDirectory,'Ephys', 
                                          database.Experiments.loc[0],
                                            'alignment.nidaq.npy'))
    
    nt = (nt * alignmentNidaq[1]) + alignmentNidaq[0]
except Exception:
    print('Could not load nidaq alignment params')
    print(traceback.format_exc())


# Gets the sparse noise file snf the props file (with the experimental
# details) for mapping RFs.
sparseFile = glob.glob(os.path.join(di, "SparseNoise*"))
propsFile = glob.glob(os.path.join(di, "props*.csv"))
propTitles = np.loadtxt(
    propsFile[0], dtype=str, delimiter=",", ndmin=2).T

if (propTitles[0] == "Spont") | (len(sparseFile) != 0): #TODO: check why this is implemented
    sparseNoise = True
    
try:
    # Gets the photodiode data.
    photodiode = nidaqData[:, nidaqChans.index("photodiode")].reshape(-1,1)  #TODO: chech this reshape
                # Gets the frames where photodiode changes are detected. 
                # fs=1 always for ephys, then index in the corrected time
    indexPhotodiode = detect_photodiode_changes(
                    photodiode, 
                    plot=pops["plot"],
                    fs=1)
    
    #TODO: check if stimulus extraction is correct here?

    changesPhotodiode = nt[indexPhotodiode.astype(int)]
except:
    print(traceback.format_exc())
    pass    


#%% Eplore stimulus info and make corrections (Gratings, Oddballs)   
stim_info = process_stimulus(propTitles, di, changesPhotodiode)

stim_info['gratings.contrast.npy'] = stim_info['gratings.contrast.npy'][5:]
stim_info['gratings.direction.npy'] = stim_info['gratings.direction.npy'][5:]
stim_info['gratings.spatialF.npy'] = stim_info['gratings.spatialF.npy'][5:]
stim_info['gratings.temporalF.npy'] = stim_info['gratings.temporalF.npy'][5:]

correct_start = stim_info['gratings.endTime.npy']
correct_end = stim_info['gratings.startTime.npy'][1:]

stim_info['gratings.startTime.npy'] = correct_start
stim_info['gratings.endTime.npy'] = correct_end


# plot photodiode and stimulus presentation corrected
plt.figure()
plt.plot(nt, nidaqData[:, nidaqChans.index("photodiode")], c='k', alpha = 0.5)
plt.plot(stim_info['gratings.startTime.npy'], 
         np.ones(len(stim_info['gratings.startTime.npy']))*0.015,
         'g*', label='stim start')

plt.plot(stim_info['gratings.endTime.npy'],
         np.ones(len(stim_info['gratings.endTime.npy']))*0.014, 
         'k*', label = 'stim end' )
plt.legend()
plt.show()    

#%% Save corrections






#%% Discard trials based on index
# photoDelIndex = [17,18]
# photoTimes = np.delete(photoTimes, photoDelIndex)

if props.columns.str.contains('Low').any():
    stimuli_start = photoTimes[:-1]
    stimuli_end = photoTimes[1:]
    
    stimuli_contrast = np.tile([0.05, 0.175], 10)[:20]
    
else: #(gratings with gray ISI)
    stim_start_index = np.arange(0,len(photoTimes),2)
    stimuli_end_index = stim_start_index + 1

    # stim_start_index = np.delete(stim_start_index, photoDelInd)
    # stimuli_end_index = np.delete(stimuli_end_index, photoDelInd)
    
    stimuli_start = photoTimes[stim_start_index]
    stimuli_end = photoTimes[stimuli_end_index]
    
    # stimuli_start = np.delete(stimuli_start, photoDelInd)
    # stimuli_end = np.delete(stimuli_end, photoDelInd)

    props = ['Ori','SFreq','TFreq','Contrast']

    StimProperties = get_stimulus_info(os.path.join(dirs['streams'], str(experiment)), props)

    stimuli_ori = np.array([int(prop['Ori']) for prop in StimProperties ]) 
    stimuli_contrast = np.array([float(prop['Contrast']) for prop in StimProperties ]) 
    stimuli_SFreq = np.array([float(prop['SFreq']) for prop in StimProperties ]) 
    stimuli_TFreq = np.array([float(prop['TFreq']) for prop in StimProperties ]) 
    
    #Discard trials based on index
    # index_ori = [38]
    # stimuli_ori = np.delete(stimuli_ori, index_ori)
    # stimuli_contrast = np.delete(stimuli_contrast, index_ori)
    # stimuli_SFreq = np.delete(stimuli_SFreq, index_ori)
    # stimuli_TFreq = np.delete(stimuli_TFreq, index_ori)

discard_stimuli = {
    
    'photoDelIndex': photoDelIndex,
    # 'index_orientation' : index_ori,
    
}

with open(os.path.join(dirs['streams'],'discard_stimuli.json'), 'w') as f:
    json.dump(discard_stimuli, f)

# plot photodiode and stimulus presentation for THESE experiments (not the correct way of doing it)
plt.figure()
plt.plot(nidaq_time_adjust, photodiode, c='k', alpha = 0.5)
# plt.plot(photoTimes[stim_end_byTimeLimit], np.ones(len(stim_end_byTimeLimit))*0.015, 'r*', label = 'stim end by time range' )
# plt.plot(photoTimes[stim_start_index], np.ones(len(stim_start_index))*0.015, 'g*', label = 'stim start' )
# plt.plot(photoTimes[stimuli_end_index], np.ones(len(stimuli_end_index))*0.014, 'k*', label = 'stim end' )
plt.plot(stimuli_start, np.ones(len(stimuli_start))*0.015, 'g*', label = 'stim start' )
plt.plot(stimuli_end, np.ones(len(stimuli_end))*0.014, 'k*', label = 'stim end' )
plt.legend()
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    