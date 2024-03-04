# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 08:41:43 2023
This script runs the z-registration over a selected number of files
IMPORTANT: Edit registration_defs to choose which files would be edited and changing the ops
Also remember to edit:
    define_directories - for the directories where the files are expected to be
    create_ops_boutton_registration - for the registration and detection options
    run together only sessions with the same registration options

"""
from zregister_function import *
from registration_defs import *
from joblib import Parallel, delayed
from zregister_function import *
from Data.user_defs import *
from Data.user_defs import directories_to_register
import pandas as pd
import traceback

# %%
dataEntries = directories_to_register()

for i in range(len(dataEntries)):
    try:
        run_single_registration(dataEntries.iloc[i])
    except:
        print(traceback.format_exc())


# Parallel(n_jobs=4, verbose=5, backend='threading')(
#     delayed(run_single_registration)(dataEntries.iloc[i])
#     for i in range(len(dataEntries))
# )
