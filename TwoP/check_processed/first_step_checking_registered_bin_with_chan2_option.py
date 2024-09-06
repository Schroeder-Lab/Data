# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:43:55 2023

@author: maria
"""
import numpy as np
import os
chan = 1

Drive = "E"
Subfolder = "ProcessedData"
# Subfolder = "Suite2Pprocessedfiles"
animal = "Ely"
date = "2024-07-02"

Directory = (
    ""
    + Drive
    + ":\\"
    + Subfolder
    + "\\"
    + animal
    + "\\"
    + date
    + "\\suite2p\\plane1\\"
)
ops = np.load(os.path.join(Directory, "ops.npy"), allow_pickle=True)
ops = ops.item()
frames = ops["frames_per_folder"]

all_exp = np.zeros((frames.shape[0], 1))
for n in range(frames.shape[0]):
    length = np.sum(frames[0 : n + 1])
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
if chan == 1:
    
    # chan1
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
    dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
    // Define your arrays for start and end times
    start_slices = newArray(
    {start_slices_str}); // Example: Starts at slices 1, 101, and 201
    end_slices = newArray({end_slices_str}); // Example: Ends at slices 100, 200, and 300

    // Open the bin
    run("Raw...", "open="+dir+"data.bin image=[16-bit Signed] width=512 height=512 number=120000 little-endian");


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
        saveAs("Tiff", dir + "/AVG_tiffs/AVG_Substack (" + start_slice + "-" + end_slice + ").tif");
        selectWindow("data.bin");
    }}"""

    with open(os.path.join(root_directory, "fijiScript.ijm"), "w") as f:
        f.write(code)

elif chan == 2:
    
    #chan2
    for dirpath, dirnames, filenamez in os.walk(root_directory):
        for dirname in dirnames:
    
            for dirp, dirr, filenames in os.walk(dirpath):
                # Check if the current directory name is "AVG_tiffs"
                if dirname != "AVG_tiffs_chan2":
                    # Check if "data.bin" exists in the filenames within the subdirectory
                    if any("data_chan2.bin" in filename for filename in filenames):
                        # Create the "AVG_tiffs" folder within each subdirectory
                        avg_tiffs_folder = os.path.join(
                            dirpath, dirname, "AVG_tiffs_chan2"
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
    dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
    // Define your arrays for start and end times
    start_slices = newArray(
    {start_slices_str}); // Example: Starts at slices 1, 101, and 201
    end_slices = newArray({end_slices_str}); // Example: Ends at slices 100, 200, and 300
    
    // Open the bin
    run("Raw...", "open="+dir+"data_chan2.bin image=[16-bit Signed] width=512 height=512 number=120000 little-endian");
    
    
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
        saveAs("Tiff", dir + "/AVG_tiffs_chan2/AVG_Substack (" + start_slice + "-" + end_slice + ").tif");
        selectWindow("data_chan2.bin");
    }}"""
    
    with open(os.path.join(root_directory, "fijiScript_chan2.ijm"), "w") as f:
        f.write(code)
