import matplotlib

from TwoP.runners import *
from user_defs import *

matplotlib.use('TkAgg')

# %% load directories and processing ops

# Please change the values in define_directories and create_processing_ops in
# module folder_defs.
dirs = define_directories()

csvDir = dirs["dataDefFile"]
s2pDir = dirs["preprocessedDataDir"]
zstackDir = dirs["zstackDir"]
metadataDir = dirs["metadataDir"]
pops = create_2p_processing_ops()

# %% read database

# In the file the values should be Name, Date, Zstack dir number, planes to
# ignore and save directory (if none default is wanted) and process (True,False)
database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Zstack": str,
        "IgnorePlanes": str,
        "SaveDir": str,
        "Process": bool,
    },
)


# %% run over data base
for i in range(len(database)):
    if not database.loc[i]["Process"]:
        continue

    print("reading directories" + str(database.loc[i]))
    # Get relevant directory paths.
    (
        suite2p_directory,
        zstackPath,
        metadataDirectory,
        saveDirectory,
    ) = read_csv_produce_directories(
        database.loc[i], s2pDir, zstackDir, metadataDir
    )
    # Disregard imaging planes that user wants to ignore ( e.g., fly-back plane when using piezo as z-actuator).
    ignorePlanes = np.atleast_1d(
        np.array(database.loc[i]["IgnorePlanes"]).astype(int)
    )
    # Load parameters that were used to process data with Suite2p.
    ops = get_ops_file(suite2p_directory)
    if pops["process_suite2p"]:
        print("getting piezo data")
        # Returns the movement of the piezo (in microns along depth, relative to top-most position of piezo
        # trace) aligned to onset of each frame.
        piezo = get_piezo_data(ops)
        print("processing suite2p data")
        # Call main processing function.
        process_s2p_directory(
            suite2p_directory,
            pops,
            piezo,
            zstackPath,
            saveDirectory=saveDirectory,
            ignorePlanes=ignorePlanes,
            debug=pops["debug"],
        )

    if pops["process_bonsai"]:
        print("reading bonsai data")
        process_metadata_directory(
            metadataDirectory, ops, pops, saveDirectory
        )
