# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:56:51 2024

@author: liad0
"""
from Data.Bonsai.behaviour_protocol_functions import *
import numpy as np
import pandas as pd


def process_stimulus(titles, directory, frameChanges):
    # go to first column, where the name of the protocol should appear
    results = stimulus_prcoessing_dictionary[titles[0][0]](
        directory, frameChanges)
    return results


def save_stimuli(saveDirectory, stimulusTypes, stimulusProps):
    stimulusTypes = np.array(stimulusTypes)
    stimulusProps = np.array(stimulusProps)
    uniqueType = np.unique(stimulusTypes)

    for t in stimulusTypes:
        props = stimulusProps[stimulusTypes == t]
        props_df = pd.DataFrame.from_records(props)
        fileNames = props_df.columns.values.astype(str)
        # save all filenames in the save directory
        for f in fileNames:
            np.save(os.path.join(saveDirectory, f), np.vstack(props_df[f]))
            

