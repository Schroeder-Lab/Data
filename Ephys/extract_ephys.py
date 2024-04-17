# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:56:47 2023

@author: Florencia
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import os
import glob
from Data.Bonsai.extract_data import *
import shutil


def read_ephys_meta_data(md_file):
    """
    
    FUNCTION MODIFIED FROM IBLLID.
    https://www.programcreek.com/python/?CodeExample=read+meta
    
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values

    Parameters
    ----------
    md_file : string
        path of spkikeGlx metadata file

    Returns
    -------
    d : dict
        Metadata params (keys) and recorded values during session.
        
    """
    import re
    
    with open(md_file) as fid:
        md = fid.read()
    d = {}
    for a in md.splitlines():
        k, v = a.split('=')
        # if all numbers, try to interpret the string
        if v and re.fullmatch('[0-9,.]*', v) and v.count('.') < 2:
            v = [float(val) for val in v.split(',')]
            # scalars should not be nested
            if len(v) == 1:
                v = v[0]
        # tildes in keynames removed
        d[k.replace('~', '')] = v

    return d

def obtain_sync_ephys (raw_neural_data, meta_neural_data):
    """
    
    Parameters
    ----------
    raw_neural_data : string
        path of raw Neuropixels data.
    meta_neural_data: dict
        dict with SpikeGLX output params
   
    Returns
    -------
    ephys_sync : np.array of int16
        sync square signal with values 0 or 64 
    """
    
    channels = int(meta_neural_data['nSavedChans'])
    
    data = np.fromfile(raw_neural_data, dtype = np.int16)
    data = np.reshape(data,(int(len(data)/channels),channels))

    ephys_sync = data[:,-1]
    
    return ephys_sync



def extract_limit_TTL (sync_signal):
    '''
    Determine start and end of pulses from a TTL signal. 

    Parameters
    ----------
    sync_signal : np.array()
        TTL signal. Assumes is a binary signal. 

    Returns
    -------
    output : np.array[sync_signal]
        Binary vector of 0 and 1. Elements equals to 1 represent start and end 
        of TTL pulses. Can be used for indexing (usally in time array).

    '''

    if len(np.unique(sync_signal)) != 2:
        print('TTL does not have 2 values, check the signal')

    sync_signal = np.where(sync_signal != 0, 1, sync_signal)
    output = np.zeros(sync_signal.shape)

    i=1
    while i < len(sync_signal)-1:
        if (sync_signal[i] == sync_signal[i+1] and 
            sync_signal[i] == sync_signal[i-1] and sync_signal[i] !=0):
          output[i] = 1
        i += 1
            
    output =  sync_signal - output  
    
    if sync_signal[0] == sync_signal[1]:
        output[0] = 0
    else:
        output[0] = 1      
    
    if sync_signal[-1] == sync_signal[-2]:
        output[-1] = 0
    else:
        output[-1] = 1      

    return output



def synchronise (syncTimes, ardSync, at, plot = False):

    '''
    Syncronise times from 2 acquisition streams using shared sync signal using 
    linear regression.

    VARIABLES COULD BE RENAMED TO MAKE IT MORE UNDERSTANDABLE
    
    Parameters
    ----------
    syncTimes : np.array()
        Stream used as reference, which should correspond to the faster acquisition stream.
        It is passing here the times of the TTL pulses (start and end) of ephys (to reduce data volume)
    ardSync : np.array()
        Sync signal recorded in the stream to align. Generally, arduino and nidaq sync signal.
    at : np.array()
        Time vector from the stream to align. Generally, arduino and nidaq time vector.
    plot : If True, it will generate the plots of mses and r2 used tu find the alignment point.
             
    Returns
    -------
    np.array([intercept, slope]) 
        Intercept and slope for the best fit, used to correct the times from the second data stream.
    
    '''   
        
    ard_sync_times = at[extract_limit_TTL (ardSync) == 1] 
    
    interval_ephys = np.diff(syncTimes)
    interval_arduino =  np.diff(ard_sync_times)
    mses= np.array([], dtype=np.float16) 
    r2 = np.array([], dtype=np.float16)
    indices= np.array([], dtype=np.float16) 
    big_gaps = np.where(interval_ephys > 0.15)[0]
    ind = 0
    
    for ind in big_gaps:
        
        stop = ind +100
                
        while ind < stop:    
            
                y = (interval_ephys)[ind:(len(interval_arduino) + ind)]
                
                if len(y) < len(interval_arduino):
                    break
                else:
        
                    y = y.reshape(-1, 1)
                    X = interval_arduino.reshape(-1, 1)
        
                    reg = LinearRegression().fit(X, y)
            
                    mse_ = mse(y, reg.predict(y))
                    mses = np.hstack((mses,mse_))
            
                    r2_ = r2_score(y, reg.predict(y))
                    r2 = np.hstack((r2,r2_))   
                    
                    indices = np.hstack((indices,ind))
                   
                    ind += 1 
                
    indEphysAlign = int(indices[np.where(mses == min(mses))[0][0]])       
    
    # final linear regresion
    reg_correction = LinearRegression().fit(
        ard_sync_times.reshape(-1, 1), 
        syncTimes[indEphysAlign:(len(ard_sync_times)+indEphysAlign)].reshape(-1, 1))
    slope = reg_correction.coef_[0]
    intercept = reg_correction.intercept_
    
    ard_time_adjust = (at * slope) + intercept
    
    if (plot):
        plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(mses, '.', markersize = 3)
        plt.plot(np.where(mses == min(mses)),min(mses), 'r*', label = 'Min')
        plt.legend()
        plt.title('MSE')
        
        plt.subplot(1,2,2)
        plt.plot(r2, '.', markersize = 3)
        plt.plot(np.where(r2 == max(r2)),max(r2), 'r*', label = 'Max')
        plt.legend()
        plt.title('R2 score')
        
        plt.show()
    
    return np.array([intercept, slope])



def synchronise_streams_with_ephys(dataEntry, pops, ephysDirectory,
                                  metadataDirectory, saveDirectory, tempEphysDir):
    '''
    This function should 
    1) take the ephys path and extract sync signal from lfp
    2) take the sync signals from arduino and nidaq from experiments asigned 
    3) align times, and give the parameters for correction for each experiment/
                    acquisition device
    4) save in the assigned directory the acorded files with the same structure than data
    '''
    
    lfpMetaDir = glob.glob(os.path.join(ephysDirectory,'*.lf.meta' ))[0] #ALWAYS LOAD THESE FILES IF SYNC IS TRUE   
    lfpRawDir = glob.glob(os.path.join(ephysDirectory,'*.lf.bin' ))[0]
    
    if tempEphysDir is not None:   
        
        if not os.path.exists(os.path.join(tempEphysDir, os.path.basename(lfpMetaDir))) or not (
          os.path.exists(os.path.join(tempEphysDir, os.path.basename(lfpRawDir)))):
        
            #Copy files to temp directory and use this as local
            shutil.copy(lfpMetaDir, tempEphysDir)
            shutil.copy(lfpRawDir, tempEphysDir)
        
        lfpMetaDir = os.path.join(tempEphysDir, os.path.basename(lfpMetaDir))
        lfpRawDir = os.path.join(tempEphysDir, os.path.basename(lfpRawDir))

    lfpMeta = read_ephys_meta_data(lfpMetaDir)
    lfpSync  = obtain_sync_ephys(lfpRawDir, lfpMeta)  
    lfpTime =  np.array(range(len(lfpSync)))/ lfpMeta['imSampRate'] 
    
    #Create a folder for sync ephys files (replace saveDirectory by save Epys)
    if not os.path.exists(os.path.join(saveDirectory,'Ephys')):
        saveEphys = os.path.join(saveDirectory,'Ephys')
        os.makedirs(saveEphys)
    else:
        saveEphys = os.path.join(saveDirectory,'Ephys')
                
    if not os.path.exists(os.path.join(saveEphys,'sync.times.npy')):
    
        syncTimes = np.where(extract_limit_TTL(lfpSync) == 1)[0] / lfpMeta['imSampRate']
        
        np.save(os.path.join(saveEphys,'sync.times.npy'),syncTimes) #TODO: determine where to save this file
    else:
        syncTimes = np.load(os.path.join(saveEphys,'sync.times.npy'))
                            
    if dataEntry.Experiments == 'all':
        #find experiment folders only, assuming that are all digits
        experiments  = [int(x) for x in os.listdir(metadataDirectory) if x.isdigit()]

    else:    
        experiments = [int(x) for x in dataEntry.Experiments.split(',')]                    
    
    for experiment in experiments: 
        try:
            experimentDirectory = os.path.join(metadataDirectory, str(experiment))
            
            dir_temps = os.path.join(saveEphys, str(experiment))
            
            if not os.path.isdir(dir_temps):
                os.makedirs(dir_temps)
            
            # Read streams: Arduino 
            # ALL ARDUINO EXTRACTION SHOULD BE PERFORMED WITH THIS FUNCTION EVERYWHERE
            
            ardData, ardChans, at = get_arduino_data(experimentDirectory)
            ardChans = [s.lower() for s in ardChans]
            
            if not ardChans:
                print(
                    f'''[Sync] File arduinoChannels.csv from {experimentDirectory} 
                    is empty, assuming recorded channels are: rotary1, rotary2,
                    camera1, camera2, sync'''
                    )
                ardChans = ['rotary1', 'rotary2', 'camera1', 'camera2', 'sync']
            
            
            ardSync = ardData[:,ardChans.index('sync')]
            alignmentArduino = synchronise (syncTimes, ardSync, at, plot = False)
            np.save(os.path.join(dir_temps, 'alignment.arduino.npy') , alignmentArduino)
            
            # Plot arduino sync
            try:
                ard_time_adjust = (at * alignmentArduino[1]) + alignmentArduino[0]
                ephysCut = (lfpTime > ard_time_adjust[0]-3) & (lfpTime < ard_time_adjust[3000])
                plt.figure(figsize=(15,5))
                plt.plot(lfpTime[ephysCut], lfpSync[ephysCut])
                plt.plot(ard_time_adjust[:3000], ardSync[:3000]*64, 'r', alpha = 0.5) #make an strange plot because of the last value, but the sync is fine!!
                plt.legend(['Neuropixels','Arduino'])
                plt.savefig(os.path.join(dir_temps,'sync_ard_exp_' + str(experiment)))
                plt.close()
            except UnboundLocalError:
                pass

            #Read streams: niDaq 
            try:
                nidaqData, nidaqChans, nt = get_nidaq_channels(experimentDirectory, plot=False)
    
                # make sure everything is in small letters
                nidaqChans = [s.lower() for s in nidaqChans]
            except:
                # TODO: why this line is not working? print(traceback.format_exc())  
                print('[Sync] Failed nidaq extraction. Assuming number of channels is 2')
                nidaqData, nidaqChans, nt = get_nidaq_channels(experimentDirectory,2,plot=False)
                nidaqChans = ['photodiode', 'sync']

            nidaqSync = nidaqData[:, nidaqChans.index("sync")]
            
            #!!!: check if this is correct: make the signal a TTL pulse 
            mean_signal = np.mean(nidaqSync)
            nidaqSync[nidaqSync < mean_signal] = 0
            nidaqSync[nidaqSync > mean_signal] = 1
            
            alignmentNidaq = synchronise (syncTimes, nidaqSync, nt, plot = False)
            
            np.save(os.path.join(dir_temps, 'alignment.nidaq.npy'), alignmentNidaq)
    
            #!!!: eliminate if nidaq processing looks good. Old functions (Florencia)
            # nidaqDir = glob.glob(os.path.join(experimentDirectory, "NiDaqInput*" ))[0]
    
            # nidaqData = GetNidaqChannels(nidaqDir, 2) #ch0 = photodiode, ch1 = sync
            # nidaqSync = nidaqData[:,1]
    
            # #make the signal a TTL pulse
            # mean_signal = np.mean(nidaqSync)
            # nidaqSync[nidaqSync < mean_signal] = 0
            # nidaqSync[nidaqSync > mean_signal] = 1
            
            # nt = np.array(range(len(nidaqSync)))/1000 #sample rate of niDAQ
               
            # alignmentNidaqOLD = synchronise (syncTimes, nidaqSync, nt, plot = True)
            
            # Plot nidaq sync
            try:
                nidaq_time_adjust = (nt * alignmentNidaq[1]) + alignmentNidaq[0]
                ephysCut = (lfpTime > nidaq_time_adjust[0]-3) & (lfpTime < nidaq_time_adjust[3000])
                
                plt.figure(figsize=(15,5))
                plt.plot(lfpTime[ephysCut], lfpSync[ephysCut])
                plt.plot(nidaq_time_adjust[:3000], nidaqSync[:3000]*64, 'r', alpha = 0.5) 
                plt.legend(['Neuropixels','niDaq'])
                plt.savefig(os.path.join(dir_temps,'sync_nidaq_exp_' + str(experiment)))
                plt.close()
            except UnboundLocalError:
                pass
        except Exception:
            print(f'''Error syncronising streams from {experimentDirectory}. 
                  Move to next experiment.''')