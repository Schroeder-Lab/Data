# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:39:57 2022

@author: LABadmin
"""

"""Runner functions"""


from joblib import Parallel, delayed
import numpy as np
import time
import traceback
import io
import os
import cv2
import skimage.io
import glob
import pickle
import scipy as sp
import warnings
from copy import deepcopy

from Data.Bonsai.extract_data import *
from Data.Ephys.user_defs_ephys import create_ephys_processing_ops
from Data.Ephys.extract_ephys import *

def process_metadata_directory_ephys(
    bonsai_dir, ops=None, pops=create_ephys_processing_ops(), 
    preprocessedDirectory=None, saveDirectory=None 
):
    """

    Processes all the metadata obtained. Assumes the metadata was recorded with
    two separated devices (in our case a niDaq and an Arduino). The niDaq was
    used to record the photodiode changes and a sync signal. 
    The Arduino was used to record the wheel movement, the camera frame times 
    and sync signal.

    The metadata processed and/or reorganised here includes:
    - the times of wheel movement and camera frames
    - sparse noise metadata: start + end times and maps
    - retinal classification metadata: start + end times and stim details
    - circles metadata: start + end times and stim details
    - gratings metadata: start + end times and stim details
    - velocity of the wheel (= mouse running velocity)

    Please have a look at the Bonsai files for the specific details for each
    experiment type.


    Parameters
    ----------
    bonsai_dir : str
        The directory where the metadata is saved.
        
    ops : dict
        For now, it only includes a list to the experiment directories to 
        analyse. But can be expanded to other params e.g. types of probes.
        
    pops : dict [2], optional
        The dictionary with all the processing infomration needed. Refer to the
        function create_processing_ops in user_defs for a more in depth
        description.
        
    preprocessedDirectory : str, optional
        Defaut directory where the processed data will be saved, determined in
        read_csv_produce_directories_ephys. If saveDir in proprocess.csv is
        empty, preprocessed data will be saved in preprocessedDataDir, defined
        in user_defs_ephys. The default is None.
        
    saveDirectory : str, optional
        Directory where the processed data will be saved.If saveDir 
        in proprocess.csv is empty, preprocessed data will be saved in 
        preprocessedDataDir, defined in user_defs_ephys. Otherwise, every output
        will be saved in saveDir from preprocess.csv. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    metadataDirectory_dirList = ops['Experiments']
        
    # Prepares the lists of outputs.

    # Recordings times, rotary encoder times, camera times.
    
    # frameTimes = [] delete probably
    wheelTimes = []
    faceTimes = []
    bodyTimes = []

    # The velocity given by the rotary encoder information.
    velocity = []

    # Stimulus information    
    stimulusProps = []
    stimulusTypes = []

    for dInd, di in enumerate(metadataDirectory_dirList):
        sparseNoise = False
        
        print(f"Directory: {di}")
        if len(os.listdir(di)) == 0:
            continue
        # Moves on if not a directory (even though ideally all should be a dir).
        # if (not(os.path.isdir(di))):
        #     continue
        expDir = os.path.split(di)[-1]
                    
        
        # Load metadata files
        # 1) Load nidaq data 
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
            alignmentNidaq = np.load(os.path.join(preprocessedDirectory,'Ephys', expDir,
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
            propsFile[0], dtype=str, delimiter=",", ndmin=2
        ).T

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
                            upThreshold=0.1,
                            downThreshold=0.2,
                            fs=1,
                            waitTime=8000)

            changesPhotodiode = nt[indexPhotodiode.astype(int)]
        except:
            print(traceback.format_exc())
            pass

        try:
            # process stimuli
            if propTitles[0][0] == 'Spont':
                stimulusResults = {"darkScreen.intervals": [nt[0], nt[-1]]}
            else:
                stimulusResults = process_stimulus(propTitles, di, changesPhotodiode)
            stimulusProps.append(stimulusResults)
            
            if propTitles[0][0] == 'Oddball':
                stimulusTypes.append('Gratings')
            else:    
                stimulusTypes.append(propTitles[0][0])
        except:
            print("Error in stimulus processing in directory: " + di)
            print(traceback.format_exc())


        # 2) Load arduino data 
        try:
            ardData, ardChans, at = get_arduino_data(di)

            # make sure everything is in small letters
            ardChans = [s.lower() for s in ardChans]
                
            if not ardChans:
                print(
                    f'''[Process] File arduinoChannels.csv from {di} 
                    is empty, assuming recorded channels are: rotary1, rotary2,
                    camera1, camera2, sync'''
                    )
                ardChans = ['rotary1', 'rotary2', 'camera1', 'camera2', 'sync']
                
            #Load arduino alignment and correct times
            try:
                alignmentArduino = np.load(os.path.join(preprocessedDirectory, 'Ephys', expDir,
                                                        'alignment.arduino.npy'))
                at = (at * alignmentArduino[1]) + alignmentArduino[0]
            except:
                print('Could not load arduino alignment params from {di}')
                print(traceback.format_exc())
    
            try:  #TODO: think if this indentation level make sense (2P is inside arduino loading)

                # TODO: Check if this records moving forward in ephys.
                movement1 = ardData[:, ardChans.index("rotary1")]
                # Gets the (assumed to be) backward movement.
                movement2 = ardData[:, ardChans.index("rotary2")]
                
                # Gets the wheel velocity in cm/s and the distance travelled in cm
    
                v, d = detect_wheel_move(movement1, movement2, at)
                # Adds the wheel times to the wheelTimes list.
                wheelTimes.append(at) #TODO: check this with Liad
                
                #Before was: 
                #wheelTimes.append(at + lastFrame)
                # Adds the velocity to the velocity list.
                velocity.append(v)
            except:
                    print("Error in wheel processing in directory: " + di)
                    print(traceback.format_exc())
            
            try: 
                # Gets the (assumed to be) face camera data.
                camera1 = deepcopy(ardData[:, ardChans.index("camera1")])
                
                # Convert TTL to binary signal
                meanCam1 = np.mean(camera1)
                camera1[camera1 < meanCam1] = 0
                camera1[camera1 > meanCam1] = 1
                
                # Gets the (assumed to be) body camera data.
                camera2 = deepcopy(ardData[:, ardChans.index("camera2")])
                
                # Convert TTL to binary signal
                meanCam2 = np.mean(camera2)
                camera2[camera2 < meanCam2] = 0
                camera2[camera2 > meanCam2] = 1
                
                #Extract TTL pulses
                cam1pulses = extract_limit_TTL(camera1) 
                cam2pulses = extract_limit_TTL(camera2)
                
                cam1times = at[cam1pulses.astype(bool)][0::2]
                cam2times = at[cam2pulses.astype(bool)][0::2]

                # Get actual video data
                vfile = glob.glob(os.path.join(di, "Video*.avi"))[0]  # eye
                video1 = cv2.VideoCapture(vfile)
                
                vfile =  glob.glob(os.path.join(di, "VideoBottom*.avi"))[0]  # body
                video2 = cv2.VideoCapture(vfile)

                #TODO: detect if there is an interval at the begining of session/ corrupted videos?

                # number of frames
                nframes1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
                nframes2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

                # 1) chek if the difference in pulses or TTL is 1 or 0, in case of one discard the last element 
                # 2) Higher diference than 1, Check manually what happened, pass
                # 3) 
    
                if len(cam1times) < nframes1:
                    if (nframes1 - len(cam1times)) < 65:
                        if (nframes1 - len(cam1times)) == 1:
                            cam1times = np.append(cam1times, np.nan)
                            faceTimes.append(cam1times)
                        else:    
                            added = np.full((nframes1 - len(cam1times)), np.nan)
                            cam1times = np.append(added, cam1times)
                            faceTimes.append(cam1times)
                            print(f'''Extreme correction performed (FG003 cases).''')
                else:
                    if (len(cam1times) - nframes1) == 1:
                        cam1times = cam1times[:-1]
                        faceTimes.append(cam1times)
                    elif (len(cam1times) - nframes1) == 0:
                        faceTimes.append(cam1times)
                    else:
                        print(f'''More than 1 extra TTL pulses than frames in face 
                              camera from {di}. Check data.''')
                
                if len(cam2times) < nframes2:
                    if (nframes2 - len(cam2times)) <65:
                        if (nframes2 - len(cam2times)) <= 2:
                            added = np.full((nframes2 - len(cam2times)), np.nan)
                            cam2times = np.append(cam2times, added)
                            bodyTimes.append(cam2times)
                        else:         
                            added = np.full((nframes2 - len(cam2times)-1), np.nan)
                            cam2times = np.append(added, cam2times)
                            cam2times = np.append(cam2times, np.nan)
                            bodyTimes.append(cam2times)
                            print(f'''Extreme correction performed (FG003 cases).''')
                else:
                    if (len(cam2times) - nframes2) == 1:
                        cam2times = cam2times[:-1]
                        bodyTimes.append(cam2times)
                    elif (len(cam2times) - nframes2) == 0:
                        bodyTimes.append(cam2times)
                    else:
                        print(f'''More than 1 extra TTL pulses than frames in body 
                              camera from {di}. Check data.''')
                        # Adds the face times to the faceTimes list.
                        # faceTimes.append(cam1times)
                        # Adds the body times to the bodyTimes list.
                        # bodyTimes.append(cam2times)
                   
                        # except:
                        #     print("Error in arduino processing in directory: " + di)
                        #     print(traceback.format_exc())
            except:
                    print("Error in camera processing in directory: " + di)
                    print(traceback.format_exc())
        except:
            print("Error in arduino processing in directory: " + di)
            print(traceback.format_exc())

    # concatante stimuli and save them
    save_stimuli(saveDirectory, stimulusTypes, stimulusProps)

    if len(wheelTimes) > 0:
        np.save(
            os.path.join(saveDirectory, "wheel.timestamps.npy"),
            np.hstack(wheelTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "wheel.velocity.npy"),
            np.hstack(velocity).reshape(-1, 1),
        )
    if (len(faceTimes) > 0):
        np.save(
            os.path.join(saveDirectory, "eye.timestamps.npy"),
            np.hstack(faceTimes).reshape(-1, 1),
        )
    if (len(bodyTimes) > 0):
        np.save(
            os.path.join(saveDirectory, "body.timestamps.npy"),
            np.hstack(bodyTimes).reshape(-1, 1),
        )



def read_csv_produce_directories_ephys(dataEntry, metadataDir, preprocessedDataDir):
    """
    Gets all the base directories (suite2p, z Stack, metadata, save directory)
    and composes these directories for each experiment.


    Parameters
    ----------
    dataEntry : pandas DataFrame [amount of experiments, 6]
        The data from the preprocess.csv file in a pandas dataframe.
        This should have been created in the main_preprocess file; assumes
        these columns are included:
            - Name
            - Date
            - Experiments
            - SaveDir
            - Sync
            - Process
    
        
    metadataDir : string
        Filepath to the metadata directory.For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.

    Returns
    -------
    ephysDirectory : string
        Filepath to raw .bin and .meta ap and lfp ephys files 
    
    metadataDirectory : string [metadataDir\Animal\Date]
        The concatenated metadata directory.
    
    preprocessedDirectory : string
        Preprocessed directory in server.
        
    saveDirectory : string [SaveDir from dataEntry or ]
        The save directory where all the processed files are saved. If not
        specified, will be saved in preprocessedDirectory.

    """
    # The data from each  dataEntry column is placed into variables.
    name = dataEntry.Name
    date = dataEntry.Date
    experiments = dataEntry.Experiments
    saveDirectory = dataEntry.SaveDir
    sync = dataEntry.Sync
    process = dataEntry.Process

    # Joins ephys directory with the name and the date.
    ephysDirectory = os.path.join(metadataDir, name, date, f'{name}_{date}_g0',
                                  f'{name}_{date}_g0_imec0')

    # If this path doesn't exist, returns a ValueError.
    if not os.path.isdir(ephysDirectory):
        raise ValueError(
            "Ephys directory " + ephysDirectory + " was not found."
        )
  
    # Joins ephys directory with the name and the date.
    metadataDirectory = os.path.join(metadataDir, name, date)

    # If metadata directory does not exist, returns this ValueError.
    if not os.path.isdir(metadataDirectory):
        raise ValueError(
            "metadata directory " + metadataDirectory + "was not found."
        )

    if not type(saveDirectory) is str:
        
        saveDirectory = os.path.join(preprocessedDataDir, name, date)
        preprocessedDirectory = saveDirectory
        
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)
       
    else:
        
        preprocessedDirectory =  os.path.join(preprocessedDataDir, name, date)
        
        saveDirectory = os.path.join(saveDirectory, "PreprocessedFiles")
        
        if not os.path.isdir(saveDirectory):
            os.makedirs(saveDirectory)     


    return ephysDirectory, metadataDirectory, preprocessedDirectory, saveDirectory


