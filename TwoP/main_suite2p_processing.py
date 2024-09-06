# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:59:08 2024

@author: maria
"""

import time
import numpy as np
from suite2p.registration import register, rigid, bidiphase
from suite2p import io
from suite2p import default_ops
from tifffile import imread
import matplotlib.pyplot as plt
from natsort import natsorted
from suite2p.registration import utils, rigid
from suite2p import run_s2p

import contextlib
from Data.TwoP.runners import read_directory_dictionary
from suite2p.io import tiff_to_binary, BinaryFile  # BinaryRWFile
from suite2p.io.utils import init_ops
import glob
from os import path

import os
import shutil

from Data.user_defs_with_neurons import *
from Data.user_defs_with_neurons import directories_to_register_neurons
import pandas as pd


dataEntries = directories_to_register_neurons()
for i in range(len(dataEntries)):
    defs = define_directories()
    s2pDir = defs["metadataDir"]
    saveDir = defs["preprocessedDataDir"]
    filePath = read_directory_dictionary(dataEntries.iloc[i], s2pDir)
    ops = create_ops_neuron_registration(filePath, saveDir=saveDir)
    
    run_s2p(ops=ops)