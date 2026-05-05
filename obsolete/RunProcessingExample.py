# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:22:36 2022

@author: LABadmin
"""
import os
import suite2p
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp


# s2pDirectory = 'D:\\Suite2Pprocessedfiles\\Hedes\\2022-03-23\\suite2p\\plane1\\'
s2pDirectory = "D:\\Suite2Pprocessedfiles\\SS113\\2022-07-11\\suite2p\\plane1\\"
# zstackPath = 'Z:\\RawData\\SS113\\2022-07-11\\4\\file_00001_00001.tif'
metadataDirectory = "Z:\\RawData\\SS113\\2022-07-11\\1\\"
saveDirectory = os.path.join(s2ps2pDirectory, "PreprocessedFiles")

if not os.path.isdir(saveDirectory):
    os.makedirs(saveDirectory)
#%% Load suite2p outputs
F = np.load(os.path.join(s2pDirectory, "F.npy"), allow_pickle=True).T
N = np.load(os.path.join(s2pDirectory, "Fneu.npy")).T
isCell = np.load(os.path.join(s2pDirectory, "iscell.npy")).T
stat = np.load(os.path.join(s2pDirectory, "stat.npy"), allow_pickle=True)
ops = np.load(os.path.join(s2pDirectory, "ops.npy"), allow_pickle=True).item()
processing_metadata = {}

fs = ops["fs"]
F = F[:, isCell[0, :].astype(bool)]
N = N[:, isCell[0, :].astype(bool)]
Fc, regPars, F_binValues, N_binValues = correct_neuropil(F, N)
F0 = get_F0(Fc, fs)
dF = get_delta_F_over_F(Fc, F0)
# TODO: save
# np.save(os.path.join(saveDirectory,'calcium.dff.npy'),dF)
#%% get Piezo Trace

nidaq, nt = get_nidaq_channels(metadataDirectory, None, plot=False)
frameclock = nidaq[:, 1]
frames = assign_frame_time(frameclock, plot=False)
piezo = nidaq[:, -2].copy()
planePiezo = get_piezo_trace_for_plane(piezo, plane=1, maxDepth=10)

#%%Create the z-stack with a piezo angle
refImg = ops["refImg"][1]
zstack = register_zstack(zstackPath, spacing=1, piezo=planePiezo, target_image=refImg)
io.imsave(os.path.join(saveDirectory, "zstackAngle.tif"), zstack)
#%%
zstack = io.imread("Z:\\RawData\\SS113\\2022-07-11\\4\\file_00001_00001_angled.tif")
#%% Retrieve z-profiles and retrieve corrected F
zprofiles = extract_zprofiles(
    s2pDirectory,
    zstack,
    neuropil_correction=regPars[1, :],
    metadata=processing_metadata,
    smooting_factor=2,
)
zTrace = np.argmax(ops["zcorr"], 0)
Fcz = correct_zmotion(dF, zprofiles, zTrace)

# TODO: save registered z-stack
np.save(os.path.join(saveDirectory, "roi.zstack.npy"), zprofiles)
np.save(
    os.path.join(saveDirectory, "roi.zstack_raw.npy"),
    processing_metadata["zprofiles_raw"],
)
np.save(
    os.path.join(saveDirectory, "roi.zstack.npy"),
    processing_metadata["zprofiles_neuropil"],
)

#%%
n = 10
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(F_corr[:, n], "k--")
ax[0].plot(F_zcorrected[:, n], "r")
ax[1].plot(zTrace)
#%%
n = 10
F = np.load(os.path.join(s2pDirectory, "F.npy")).T
plt.plot(F[:, n], "r")
plt.plot(F_corr[:, n], "k")
plt.plot(N[:, n], "b")

#%%
plt.close("all")
stat = np.load(os.path.join(s2pDirectory, "stat.npy"), allow_pickle=True)
ops = np.load(os.path.join(s2pDirectory, "ops.npy"), allow_pickle=True).item()
refImg = ops["refImg"][0]
n = 4


for n in range(zprofiles.shape[1]):

    im = np.zeros((ops["Ly"], ops["Lx"]))
    # im = refImg
    ypix = stat[n]["ypix"]  # [~stat[n]['overlap']]

    xpix = stat[n]["xpix"]  # [~stat[n]['overlap']]
    im[ypix, xpix] = 1
    # im[nypix,nxpix] = 0
    plt.figure()
    r = refImg.copy()
    r[im.astype(bool)] = -10000
    plt.imshow(r)
    plt.figure()
    plt.plot(zprofiles[:, n], "k")
    # plt.vlines([np.min(zTrace),np.max(zTrace)],np.min(zprofiles[:,n])-100,np.max(zprofiles[:,n])+100,'r')
    plt.vlines(
        [np.nanpercentile(zTrace, 2.5), np.nanpercentile(zTrace, 97.5)],
        np.min(zprofiles[:, n]) - 100,
        np.max(zprofiles[:, n]) + 100,
        "r",
    )
    plt.ylim(np.min(zprofiles[:, n]), np.max(zprofiles[:, n]))
    plt.savefig(s2pDirectory + "Zprofiles\\neuron" + str(n) + "_corrected.png")
    plt.close("all")
    plt.figure()
    plt.plot(zprofilesC[:, n], "k")
    plt.plot(neuropil[:, n], "b")
    # plt.vlines([np.min(zTrace),np.max(zTrace)],np.min(zprofilesC[:,n])-100,np.max(zprofilesC[:,n])+100,'r')
    plt.vlines(
        [np.nanpercentile(zTrace, 2.5), np.nanpercentile(zTrace, 97.5)],
        np.min(zprofilesC[:, n]) - 100,
        np.max(zprofilesC[:, n]) + 100,
        "r",
    )
    plt.savefig(s2pDirectory + "Zprofiles\\neuron" + str(n) + "_raw.png")
    plt.close("all")
#%% Hunt for the ROI
plt.close("all")
stat = np.load(os.path.join(s2pDirectory, "stat.npy"), allow_pickle=True)
stat = stat[isCell[0, :].astype(bool)]
ops = np.load(os.path.join(s2pDirectory, "ops.npy"), allow_pickle=True).item()
refImg = ops["refImg"][1]


n = 0
im = np.zeros((ops["Ly"], ops["Lx"]))
# im = refImg
ypix = stat[n]["ypix"][~stat[n]["overlap"]]

xpix = stat[n]["xpix"][~stat[n]["overlap"]]
im[ypix, xpix] = 1
# im[nypix,nxpix] = 0
plt.figure()
r = refImg.copy()
r[im.astype(bool)] = -10000
plt.imshow(im, cmap="bone")

#%% Fitting Tryouts
plt.close("all")
plt.plot(zprofilesC[:, n], "k--")
n = 16


def _moffat(r, B, A, r0, alpha, beta):
    return B + A * (1 + (((r - r0) ** 2) / alpha ** 2)) ** (-beta)


def _gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


z = zprofilesC[:, n]
zprofilesC[:, n]
z /= np.max(z)
z -= np.min(z)

res, _ = optimize.curve_fit(
    _moffat,
    np.arange(0, 101),
    z,
    p0=(0, 1, 50, 1, 1),
    bounds=(
        [0, 0, -np.inf, -np.inf, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
    ),
)


plt.plot(_moffat(np.arange(0, 101), *res), "r")

#%% Load saved data
calSig = np.load(os.path.join(saveDirectory, "calcium.dff.npy"))
frameTimes = np.load(os.path.join(saveDirectory, "calcium.frameTimes.npy"))
st = np.load(os.path.join(saveDirectory, "events.st.npy"))
eventType = np.load(os.path.join(saveDirectory, "events.type.npy"))
planes = np.load(os.path.join(saveDirectory, "calcium.planes.npy"))
delays = np.load(os.path.join(saveDirectory, "planes.timeDelta.npy"))
sparse = np.load(os.path.join(saveDirectory, "sparse.sparseMap.npy"))
gst = st[eventType == "grating"]

ca, t = GetCalciumAligned(
    calSig,
    frameTimes.reshape(-1, 1),
    gst,
    np.array([-0.05, 0.16]).reshape(-1, 1).T,
    planes.reshape(-1, 1),
    delays.reshape(1, -1),
)

#%%
n = 32
mapz = np.zeros((8, 9))
for x in range(8):
    for y in range(9):
        inds = np.where(sparse[:, x, y] != 0.5)[0]
        sig = ca[:, :, n][:, inds]
        mapz[x, y] = np.mean(np.nanmean(sig[t >= 0, :], 1))
