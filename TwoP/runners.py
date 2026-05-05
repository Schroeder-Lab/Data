import matplotlib

from Bonsai.extract_data import *
from TwoP.general import *


matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 6})


def read_directory_dictionary(dataEntry, s2pDirectory):
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
            - Zstack
            - IgnorePlanes
            - SaveDir
            - Process
    s2pDir : string
        Filepath to the Suite2P processed folder. For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.
    zstackDir : string
        Filepath to the Z stack.For more details on what this should contain
        please look at the define_directories function definition in
        user_defs.
    metadataDir : string
        Filepath to the metadata directory.For more details on what this
        should contain please look at the define_directories function
        definition in user_defs.

    Returns
    -------
    s2pDirectory : string [s2pr\Animal\Date\suite2p]
        The concatenated Suite2P directory.
    zstackPath : string [zstackDir\Animal\Date\Z stack value from
        dataEntry\Z_stack_file.tif]
        The concatenated Z stack directory.
    metadataDirectory : string [metadataDir\Animal\Date]
        The concatenated metadata directory.
    saveDirectory : string [SaveDir from dataEntry or ]
        The save directory where all the processed files are saved. If not
        specified, will be saved in the suite2p folder.

    """
    # The data from each  dataEntry column is placed into variables.
    name = dataEntry.Name
    date = dataEntry.Date
    experiments = np.atleast_1d(dataEntry.Experiments)

    # Joins suite2p directory with the name and the date.
    s2pDirectory = os.path.join(s2pDirectory, name, date)

    # If this path doesn't exist, returns a ValueError.
    if not os.path.exists(s2pDirectory):
        raise ValueError(
            "suite 2p directory " + s2pDirectory + "was not found."
        )

    dataPaths = [s2pDirectory]
    for i, e in enumerate(experiments):
        dataPaths.append(os.path.join(s2pDirectory, str(e)))

    return dataPaths


