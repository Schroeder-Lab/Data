# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:56:09 2023

@author: Liad

Print out sparse noise results for an entire image to roughly know the 
receptive field.

! IMPORTANT ! 

Remember to set create_ops_boutton_registration for the options that suit your recording (num of pixels, num of channels etc)
Set  get_sparsenoise_info with only one session and the save directory 

"""


from Data.TwoP.runners import read_directory_dictionary
from Data.user_defs import (
    define_directories,
    create_sparse_registration,
    get_sparsenoise_info,
)
import numpy as np
import pandas as pd
import time
import os
import shutil
from suite2p.registration import register, rigid, bidiphase
from suite2p.io import tiff_to_binary, BinaryFile
from suite2p import io
from suite2p import default_ops
from tifffile import imread
import matplotlib.pyplot as plt
from natsort import natsorted
import imp
from suite2p import default_ops
from suite2p.registration import utils, rigid
from suite2p import run_s2p
from Data.TwoP.registration_defs import *
from Data.TwoP.runners import *


sparseSession, tmpSave = get_sparsenoise_info()

if not os.path.isdir(tmpSave):
    os.makedirs(tmpSave)


defs = define_directories()
s2pDir = defs["metadataDir"]
filePath = read_directory_dictionary(sparseSession, s2pDir)
# REMEMB
ops = create_sparse_registration(filePath, tmpSave)
ops["save_path"] = tmpSave

# %%
# do registration
# convert tiffs to binaries
# if "save_folder" not in ops or len(ops["save_folder"]) == 0:
#     ops["save_folder"] = tmpSave +"\\suite2p"
# save_folder = os.path.join(ops["save_path"], ops["save_folder"])
# os.makedirs(save_folder, exist_ok=True)

#%%
ops = tiff_to_binary(ops)


# # get plane folders
# plane_folders = natsorted(
#     [
#         f.path
#         for f in os.scandir(save_folder)
#         if f.is_dir() and f.name[:5] == "plane"
#     ]
# )
# ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
# nplanes = len(ops_paths)

# # compute reference image
# refImgs = []

# for ipl, ops_path in enumerate(ops_paths):
#     if ipl in ops["ignore_flyback"]:
#         print(">>>> skipping flyback PLANE", ipl)
#         continue

#     ops = np.load(ops_path, allow_pickle=True).item()
#     align_by_chan2 = ops["functional_chan"] != ops["align_by_chan"]
#     raw = ops["keep_movie_raw"]
#     reg_file = ops["reg_file"]
#     raw_file = ops.get("raw_file", 0) if raw else reg_file
#     if ops["nchannels"] > 1:
#         reg_file_chan2 = ops["reg_file_chan2"]
#         raw_file_chan2 = (
#             ops.get("raw_file_chan2", 0) if raw else reg_file_chan2
#         )
#     else:
#         reg_file_chan2 = reg_file
#         raw_file_chan2 = reg_file

#     align_file = reg_file_chan2 if align_by_chan2 else reg_file
#     align_file_raw = raw_file_chan2 if align_by_chan2 else raw_file
#     Ly, Lx = ops["Ly"], ops["Lx"]

#     # M:this part of the code above just does registration etc (what is done with the GUI usually)
#     # grab frames
#     with BinaryRWFile(Ly=Ly, Lx=Lx, filename=align_file_raw) as f_align_in:
#         n_frames = f_align_in.shape[0]
#         frames = f_align_in[
#             np.linspace(
#                 0,
#                 n_frames,
#                 1 + np.minimum(ops["nimg_init"], n_frames),
#                 dtype=int,
#             )[:-1]
#         ]

#     # M: this is done to adjust bidirectional shift occuring due to line scanning
#     # compute bidiphase shift
#     if (
#         ops["do_bidiphase"]
#         and ops["bidiphase"] == 0
#         and not ops["bidi_corrected"]
#     ):
#         bidiphase = bidiphase.compute(frames)
#         print(
#             "NOTE: estimated bidiphase offset from data: %d pixels" % bidiphase
#         )
#         ops["bidiphase"] = bidiphase
#         # shift frames
#         if bidiphase != 0:
#             bidiphase.shift(frames, int(ops["bidiphase"]))
#     else:
#         bidiphase = 0

#     # compute reference image
#     refImgs.append(register.compute_reference(frames))

# # align reference frames to each other
# frames = np.array(refImgs).copy()
# for frame in frames:
#     rmin, rmax = np.int16(np.percentile(frame, 1)), np.int16(
#         np.percentile(frame, 99)
#     )
#     frame[:] = np.clip(frame, rmin, rmax)

# refImg = frames.mean(axis=0)
# # M: the below section is just the usual xy registration
# niter = 8
# for iter in range(0, niter):
#     # rigid registration
#     ymax, xmax, cmax = rigid.phasecorr(
#         data=rigid.apply_masks(
#             frames,
#             *rigid.compute_masks(
#                 refImg=refImg,
#                 maskSlope=ops["spatial_taper"]
#                 if ops["1Preg"]
#                 else 3 * ops["smooth_sigma"],
#             ),
#         ),
#         cfRefImg=rigid.phasecorr_reference(
#             refImg=refImg, smooth_sigma=ops["smooth_sigma"]
#         ),
#         maxregshift=ops["maxregshift"],
#         smooth_sigma_time=ops["smooth_sigma_time"],
#     )
#     dys = np.zeros(len(frames), "int")
#     dxs = np.zeros(len(frames), "int")
#     for i, (frame, dy, dx) in enumerate(zip(frames, ymax, xmax)):
#         frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
#         dys[i] = dy
#         dxs[i] = dx

# print("shifts of reference images: (y,x) = ", dys, dxs)

# # frames = smooth_reference_stack(frames, ops)

# refImgs = list(frames)

# # register and choose the best plane match at each time point,
# # in accordance with the reference image of each plane

# imp.reload(utils)
# imp.reload(rigid)
# imp.reload(register)

# ops["refImg"] = refImgs
# ops_paths_clean = np.delete(ops_paths, ops["ignore_flyback"])
# # Get the correlation between the reference images

# for ipl, ops_path in enumerate(ops_paths):
#     if ipl in ops["ignore_flyback"]:
#         print(">>>> skipping flyback PLANE", ipl)
#         continue
#     else:
#         print(">>>> registering PLANE", ipl)
#     ops = np.load(ops_path, allow_pickle=True).item()
#     ops["nonrigid"] = True
#     ops["block_size"] = [128, 128]
#     ops = register.register_binary(ops, refImg=refImgs)
#     np.save(ops["ops_path"], ops)


# %% load bonsai stuff

plane = sparseSession["Plane"]
readDir = os.path.join(ops["save_path0"], "suite2p", f"plane{plane}")
ops = np.load(os.path.join(readDir, "ops.npy"), allow_pickle=True).item()
process_metadata_directory(
    ops["data_path"], ops, saveDirectory=ops["save_path0"]
)

# %%
ts = np.load(os.path.join(ops["save_path0"], "calcium.timestamps.npy"))
st = np.load(os.path.join(ops["save_path0"], "sparse.startTime.npy"))
smap = np.load(os.path.join(ops["save_path0"], "sparse.map.npy"))
edges = np.load(os.path.join(ops["save_path0"], "sparseExp.edges.npy"))


readDir = os.path.join(ops["save_path0"], "suite2p", f"plane{plane}")
binPath = os.path.join(readDir, "data_raw.bin")
ops = np.load(os.path.join(readDir, "ops.npy"), allow_pickle=True).item()

f = np.nanmean(np.diff(ts, axis=0))
window = [-0.3, 0.3]
wt = (window / f).astype(int)
t = np.arange(*wt)


n_frames, Ly, Lx = ops["nframes"], ops["Ly"], ops["Lx"]

averageMap = np.zeros((smap.shape[1], smap.shape[2], Ly, Lx))
# get the locked frame per map position
for y in range(smap.shape[1]):
    for x in range(smap.shape[2]):
        inds = np.where(smap[:, y, x] != 0.5)[0]
        sts = st[inds]
        maps = np.zeros((len(sts), Ly, Lx))
        for si, s in enumerate(sts):
            # find the first instance where s is after the clock time
            firstInd = np.where(s >= ts)[0][-1]
            sw = wt + firstInd
            if (np.any((t+firstInd) > n_frames)):
                continue
            with BinaryFile(
                Ly=ops["Ly"], Lx=ops["Lx"], filename=binPath, n_frames=n_frames
            ) as f_bin:
                frames = f_bin[t + firstInd]
                corrected = np.nanmean(frames[t >= 0], 0) - np.nanmean(
                    frames[t < 0], 0
                )
                maps[si, :, :] = corrected

        averageMap[y, x, :, :] = np.nanmean(maps, 0)

# %%
averageMap_ncd = sp.signal.convolve2d(
    averageMap_n[0, 0, :, :], np.ones((16, 16)), mode="same"
)
# %%
averageMap_n = averageMap - np.nanmin(averageMap)
averageMap_n = averageMap_n / np.nanmax(averageMap_n)


xEdges = np.linspace(edges[4], edges[5], averageMap.shape[1] + 1)[:, 0]
yEdges = np.linspace(edges[3], edges[2], averageMap.shape[0] + 1)[:, 0]
xps = np.round(np.abs((edges[5] - edges[4]) / averageMap.shape[1]), 1)
yps = np.round(np.abs((edges[3] - edges[2]) / averageMap.shape[0]), 1)

f, ax = plt.subplots(averageMap.shape[0], averageMap.shape[1])
f.suptitle(
    f"Response per square \n Each square is roughly {yps}X{xps} degrees\ny range: {edges[0]}-{edges[1]}, x range: {edges[2]}-{edges[3]}.\n num of squares: x: {averageMap.shape[1]}, y:{averageMap.shape[0]}"
)
maxVal = -np.inf
for y in range(averageMap.shape[0]):
    for x in range(averageMap.shape[1]):
        if (x != 0) & (x != averageMap.shape[1] - 1):
            ax[y, x].set_xticklabels([])
        if (y != 0) & (y != averageMap.shape[0] - 1):
            ax[y, x].set_yticklabels([])
        im = ax[y, x].imshow(
            sp.ndimage.gaussian_filter(averageMap_n[y, x], 8),
            extent=[xEdges[x], xEdges[x + 1], yEdges[y + 1], yEdges[y]],
            aspect="auto",
        )

        mx = np.nanmax(im.get_clim())
        maxVal = max(mx, maxVal)
# maxVal = 1
for y in range(averageMap.shape[0]):
    for x in range(averageMap.shape[1]):
        ax[y, x].get_images()[0].set_clim(0, maxVal)
plt.tight_layout()
