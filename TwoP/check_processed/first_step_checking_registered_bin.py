# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:43:55 2023

@author: maria
"""
import numpy as np
import os


dirDefs = [

    # {
    #     "Name": "Oephelia",
    #     "Date": "2023-07-21",
    #     "Experiments": [1, 2, 3, 4, 5],
    # },
    # {
    #     "Name": "Oephelia",
    #     "Date": "2023-07-26",
    #     "Experiments": [1, 2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-21",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-22",
    #     "Experiments": [1, 3, 4],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-30",
    #     "Experiments": [2, 3, 4, 5, 6, 7],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-31",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-05",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-08",
    #     "Experiments": [2, 3],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-18",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-21",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-27",
    #     "Experiments": [2, 3, 4, 5],
    # },

    # {
    #     "Name": "Memphis",
    #     "Date": "2023-10-05",
    #     "Experiments": [1, 2, 3, 4, 5, 6, 7],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-10-18",
    #     "Experiments": [1, 2, 3, 4],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-11-07",
    #     "Experiments": [1, 2, 3],
    # },

    # {
    #     "Name": "Styx",
    #     "Date": "2024-01-15",
    #     "Experiments": [2, 3, 4, 5, 6, 7, 8],
    # },
    # {
    #     "Name": "Styx",
    #     "Date": "2024-01-23",
    #     "Experiments": [2, 3, 4, 5, 6, 7, 8, 9],
    # },
    # {
    #     "Name": "Styx",
    #     "Date": "2024-01-31",
    #     "Experiments": [2, 3, 4, 5, 6, 7],
    # },
    # {
    #     "Name": "Styx",
    #     "Date": "2024-02-01",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },



    # {
    #     "Name": "Styx",
    #     "Date": "2024-02-20",
    #     "Experiments": [2, 3, 4, 5, 6, 7, 8],
    # },



    # {
    #     "Name": "Styx",
    #     "Date": "2024-02-26",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Stereopes",
    #     "Date": "2024-01-19",
    #     "Experiments": [2, 3, 4, 5, 6, 7],
    # },



    {
        "Name": "Vesta",
        "Date": "2024-05-08",
        "Experiments": [2, 3, 4, 5, 6, 7],
    },
    {
        "Name": "Vesta",
        "Date": "2024-05-14",
        "Experiments": [2, 3, 4, 5],
    },
    {
        "Name": "Vesta",
        "Date": "2024-05-15",
        "Experiments": [2, 3, 4, 5, 6],
    },
    # {
    #     "Name": "Vesta",
    #     "Date": "2024-05-17",
    #     "Experiments": [2, 3, 4, 5],
    # },
    {
        "Name": "Vesta",
        "Date": "2024-06-05",
        "Experiments": [2, 3, 4, 5, 6, 7, 8],
    },
    # {
    #     "Name": "SS123",
    #     "Date": "2024-07-31",
    #     "Experiments": [2, 3, 4],
    # },
    # {
    #     "Name": "SS123",
    #     "Date": "2024-07-08",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-21",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-22",
    #     "Experiments": [1, 3, 4],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-30",
    #     "Experiments": [2, 3, 4, 5, 6, 7],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-08-31",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-05",
    #     "Experiments": [2, 3, 4, 5, 6],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-08",
    #     "Experiments": [2, 3],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-18",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-21",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-09-27",
    #     "Experiments": [2, 3, 4, 5],
    # },




    # {
    #     "Name": "Memphis",
    #     "Date": "2023-10-05",
    #     "Experiments": [2, 3, 4, 5, 6, 7],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-10-18",
    #     "Experiments": [2, 3, 4, 5],
    # },
    # {
    #     "Name": "Memphis",
    #     "Date": "2023-11-07",
    #     "Experiments": [1, 2, 3],
    # },
    {
        "Name": "Stereopes",
        "Date": "2024-01-19",
        "Experiments": [2, 3, 4, 5, 6, 7],
    },



    {
        "Name": "Stereopes",
        "Date": "2024-01-29",
        "Experiments": [2, 4, 5, 6, 7],
    },



    {
        "Name": "Stereopes",
        "Date": "2024-02-12",
        "Experiments": [1, 2, 3, 4, 5],
    },
    {
        "Name": "Stereopes",
        "Date": "2024-02-13",
        "Experiments": [1, 2, 3, 4, 5],
    },
    {
        "Name": "Stereopes",
        "Date": "2024-02-15",
        "Experiments": [1, 2, 3, 4, 5],
    },
    {
        "Name": "Stereopes",
        "Date": "2024-02-21",
        "Experiments": [2, 3, 4, 5, 6, 7, 8],
    },
    {
        "Name": "Stereopes",
        "Date": "2024-02-28",
        "Experiments": [2, 3, 4, 5],
    },
]


for dd in dirDefs:

    Drive = "Z"
    Subfolder = "ProcessedData"
    # Subfolder = "Suite2Pprocessedfiles"
    animal = dd['Name']
    date = dd['Date']

    Directory = (
        ""
        + Drive
        + ":\\"
        + Subfolder
        + "\\"
        + animal
        + "\\"
        + date
        + "\\suite2p\\plane\\"
    )
    ops = np.load(os.path.join(Directory, "ops.npy"), allow_pickle=True)
    ops = ops.item()
    frames = ops["frames_per_folder"]
    all_exp = np.zeros((frames.shape[0], 1))
    for n in range(frames.shape[0]):
        length = np.sum(frames[0: n + 1])
        all_exp[n] = length

    experiment_num = 2

    # Initialize the first range as "0-100"
    ranges = ["0-100"]

    # Calculate the ranges based on the input array
    for num in all_exp:
        num = int(num)
        range_str_end = f"{num - 100}-{num}"
        range_str_start = f"{num}-{num+100}"
        range_list = [range_str_end, range_str_start]
        ranges.append(range_list)

    # Initialize empty arrays
    start_slices = np.array([], dtype=int)
    end_slices = np.array([], dtype=int)

    for sublist in ranges:
        if isinstance(sublist, list):
            for item in sublist:
                start, end = map(int, item.split("-"))
                start_slices = np.append(start_slices, start)
                end_slices = np.append(end_slices, end)

    # Add 1 to the start_slices array so it's 100 frames
    # (conversion from ImageJ to Python numbering)
    start_slices += 1

    # Add 1 to the beginning of start_slices array
    start_slices = np.insert(start_slices, 0, 1)

    # Add 100 to the end of end_slices array
    end_slices = np.insert(end_slices, 0, 100)

    # delete last one as it's outside of the frame numberand gives error in ImageJ
    start_slices = np.delete(start_slices, -1)
    end_slices = np.delete(end_slices, -1)

    # Print values from start_slices separated by commas
    print("Start Slices:")
    start_slices_str = ", ".join(map(str, start_slices))
    print(start_slices_str)

    # Print values from end_slices separated by commas
    print("\nEnd Slices:")
    end_slices_str = ", ".join(map(str, end_slices))
    print(end_slices_str)

    root_directory = (
        ""
        + Drive
        + ":\\"
        + Subfolder
        + "\\"
        + animal
        + "\\"
        + date
        + "\\suite2p\\"
    )

    for dirpath, dirnames, filenamez in os.walk(root_directory):
        for dirname in dirnames:

            for dirp, dirr, filenames in os.walk(dirpath):
                # Check if the current directory name is "AVG_tiffs"
                if dirname != "AVG_tiffs":
                    # Check if "data.bin" exists in the filenames within the subdirectory
                    if any("data.bin" in filename for filename in filenames):
                        # Create the "AVG_tiffs" folder within each subdirectory
                        avg_tiffs_folder = os.path.join(
                            dirpath, dirname, "AVG_tiffs"
                        )
                        if not os.path.exists(avg_tiffs_folder):
                            os.makedirs(avg_tiffs_folder, exist_ok=True)
                            print(
                                "\nAVG_tiffs folders now created in all subdirectories."
                            )

    code = f"""
    // Define your variables
    drive = "{Drive}"; // Drive letter
    animal = "{animal}"; // Animal name
    date = "{date}"; // Date
    plane = "plane1"; // Plane name
    processing_folder = "{Subfolder}";
    //processing_folder = "Suite2Pprocessedfiles";
    dir =  drive + ":/"+processing_folder+"/" + \
        animal + "/" + date + "/suite2p/" + plane + "/";
    // Define your arrays for start and end times
    start_slices = newArray(
    {start_slices_str}); // Example: Starts at slices 1, 101, and 201
    end_slices = newArray({end_slices_str}); // Example: Ends at slices 100, 200, and 300

    // Open the bin
    run("Raw...", "open="+dir+ \
        "data.bin image=[16-bit Signed] width=256 height=256 number=120000 little-endian");


    // Loop over the arrays
    for (i = 0; i < lengthOf(start_slices); i++) {{
        start_slice = start_slices[i];
        end_slice = end_slices[i];

        // Convert start and end slices to strings and create the substack using variables
        //run("Make Substack...", "slices=" +start_slice+ "-" +end_slice"");
        run("Make Substack...", "slices="+start_slice+"-"+end_slice);
        run("Z Project...", "projection=[Average Intensity]");

        // Close the substack window
        selectWindow("Substack (" + start_slice + "-" + end_slice + ")"); // Adjust the window title if needed
        close();
        saveAs("Tiff", dir + "/AVG_tiffs/AVG_Substack (" + \
               start_slice + "-" + end_slice + ").tif");
        selectWindow("data.bin");
    }}"""

    with open(os.path.join(root_directory, "fijiScript.ijm"), "w") as f:
        f.write(code)
