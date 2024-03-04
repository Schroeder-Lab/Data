"""Pre-process tiff files."""


import os
import numpy as np
import pandas as pd
import skimage
import scipy as sp
from skimage import io
from skimage import data
from skimage import metrics
from skimage.util import img_as_float
import tifftools as tt
import pandas as pd

# from pystackreg import StackReg
from suite2p.extraction.extract import extract_traces
from suite2p.extraction.masks import create_masks
from suite2p.registration.register import register_frames, compute_reference
from suite2p.registration import rigid
from Data.TwoP.preprocess_traces import correct_neuropil
from suite2p.default_ops import default_ops
from numba import jit

from Data.TwoP.preprocess_traces import zero_signal


# @jit(forceobj=True)
def _fill_plane_piezo(stack, piezoNorm, i, spacing=1):
    """
    Slants the Z stack planes according to how slanted the imaged frames are.
    This is done because the frames are acquired using a fast scanning technique
    (sawtooth scanning) which means that, along the y axis, the Z differs.
    This is different to taking the Z stack which uses normal scanning which
    means the depth along the Y axis is the same. To make sure the Z stack is
    aligned in Y to the imaged frames, the function slants the Z stack.

    Parameters
    ----------
    stack : array [z,x,y]
        the registered image stack.
    piezoNorm : array [t]
        the piezo depth normalised to the stack spacing.
    i: int
        the frame to create
    Returns
    -------
    a normalised frame.

    """
    # Normalises the piezo trace to the top depth.
    piezoNorm -= piezoNorm[0]

    # Adds the value of the current Z stack plane (equivalent to 1um/plane).
    piezoNorm += i

    # Gets the number of planes and no. of pixels along X and Y of the Z stack.
    planes = stack.shape[0]
    resolutionx = stack.shape[1]
    resolutiony = stack.shape[2]

    # interpolate
    ratio = resolutiony/len(piezoNorm)
    # f = sp.interpolate.interp1d(np.arange(piezoNorm.shape[0]), piezoNorm)
    # piezoNorm = f(np.arange(0,piezoNorm.shape[0]-1,spacing/10))
    f = sp.interpolate.interp1d(np.linspace(
        0, resolutiony-1, len(piezoNorm)), piezoNorm[:, 0])
    piezoNorm = f(np.arange(0, resolutiony))

    # Creates a variable that tells the current location in Y (in pixels).
    currPixelY = 0
    # Will contain the slanted image in the current plane.
    slantImg = np.zeros(stack.shape[1:])
    # Gets the total amount of pixels in y
    # (used to calculate how many pixels per piezo step).
    pixelsPerMoveY = np.ones(len(piezoNorm)) * resolutiony

    # Gets the number of pixels per piezo step.
    # np.round(pixelsPerMoveY / len(piezoNorm)).astype(int)
    numPixelsY = pixelsPerMoveY / len(piezoNorm)

    # Corrects in case of rounding error.
    Yerr = resolutiony-1 - sum(numPixelsY)
    numPixelsY[-1] += Yerr

    # Gets the end points (in pixels) of each piezo step.
    pixelsY = np.cumsum(numPixelsY).astype(int)
    # Creates an interpolating function based on the z stack.
    interp = sp.interpolate.RegularGridInterpolator(
        (
            np.arange(0, planes, spacing),
            np.arange(0, resolutiony),
            np.arange(0, resolutionx),
        ),
        stack,
        fill_value=None,
        bounds_error=False,
        method='nearest'
    )
    for yt in range(len(piezoNorm)):
        depth = piezoNorm[yt]

        # If beyond the depth, takes the final frame.
        # if depth > planes - 1:
        #     depth = planes - 1
        # If below the topmost frame, takes the first one.
        # if depth < 0:
        #     depth = 0

        line = interp(
            (
                depth,
                yt,
                np.arange(0, resolutionx),
            )
        )

        slantImg[int(yt), 0:resolutionx] = line
    # for d in range(len(piezoNorm)):  # For each piezo step
    #     endPointY = pixelsY[
    #         d
    #     ]  # Gets the end point for the current piezo step.
    #     depth = piezoNorm[d]  # Gets the current depth from the piezo trace.

    #     # If beyond the depth, takes the final frame.
    #     if depth > planes - 1:
    #         depth = planes - 1
    #     # If below the topmost frame, takes the first one.
    #     if depth < 0:
    #         depth = 0
    #     # For every pixel within the current piezo step.
    #     for yt in np.arange(currPixelY, endPointY):
    #         # print (depth,yt)
    #         # Determines the approximate pixel values along one y line
    #         # given the depth.
    #         line = interp(
    #             (
    #                 depth,
    #                 yt,
    #                 np.arange(0, resolutionx),
    #             )
    #         )
    #         # Appends the newly created line to the slanted image stack.
    #         slantImg[np.floor(yt), 0:resolutionx] = line
    #     # Updates the current location in y.
    #     currPixelY += numPixelsY[d]

    return slantImg


def _register_swipe(zstack, start, finish, progress):
    """
    Performs local registration by registering a plane with one plane before it
    and two planes after it. For the first plane, it only takes the two planes
    after it. For the penultimate plane it only takes one plane after it (since
    that would be the last plane).

    Parameters
    ----------
    zstack : np.ndarray [planes, y, x]
        The Z stack to register.
    start : int
        The first plane to register.
    finish : int
        the final plane to register.
    progress :  int
        the step size of the range.

    Returns
    -------
    zstack : np.ndarray [planes, x, y]
        The locally registered Z stack.

    """
    # print(str(start)+', finish:' + str(finish)+', progress:'+str(progress))
    for i in range(start, finish, progress):
        # For the first plane, the range is the 2 (first plane and one plane
        # after it).
        if i == 0:
            stackRange = range(i, i + 2)
        # For the final plane, the range is 2 (penultimate plane and last
        # plane).
        elif i == zstack.shape[0] - 1:
            stackRange = range(i - 1, i + 1)
        else:
            # For all planes in between the first and the last plane the range is
            # the plane before, the current plane and the plane after.
            stackRange = range(i - 1, i + 2)
        # Makes a small stack with the planes specified by the range above.
        miniStack = zstack[stackRange]
        # Registers the planes uisng the frame registration function from
        # suite2p by using the middle plane as a reference.
        res = register_frames(
            miniStack[1, :, :], miniStack[:, :, :].astype(np.int16)
        )
        # Appends the z stack with these locally registered planes.
        zstack[stackRange, :, :] = res[0]
    return zstack


def register_zstack_frames(zstack):
    """
    Wrapper-like function. Performs interative local registration through the
    sub-function _register_swipe.
    1. Registers from the mid plane to the top plane
    (assuming the first plane in the Z stack is the top plane).
    2. Registers from the mid plane to the bottom plane.
    3. Registers from the top to the bottom plane.

    Parameters
    ----------
    zstack : np.ndarray [planes, x, y]
        The Z stack to register.

    Returns
    -------
    zstack : np.ndarray [planes, x, y]
        The locally registered Z stack.

    """
    # Start from centre take triples and align them
    centreFrame = int(np.floor(zstack.shape[0] / 2))
    # Performs registration from mid to top plane.
    zstack = _register_swipe(zstack, centreFrame, 0, -1)
    # Performs registration from mid to bottom plane.
    zstack = _register_swipe(zstack, centreFrame, zstack.shape[0], 1)
    # Performs registration from top to bottom plane.
    zstack = _register_swipe(zstack, 0, zstack.shape[0], 1)
    return zstack


def register_stack_to_ref(zstack, refImg, ops=default_ops()):
    """
    Registers the Z stack to the reference image using the same approach
    as registering the frames to the reference image.
    All functions come from suite2p, see their docs for further information on
    the functions.

    Parameters
    ----------
    zstack : np.ndarray [planes, x, y]
        The Z stack to register.
    refImg : np.ndarray [x, y]
        The reference image (from suite2p).
    ops : dict, optional
        The ops dictionary from suite2p. The default is default_ops().

    Returns
    -------
    zstackCorrected : np.ndarray [planes, x, y]
        The corrected Z stack.

    """
    # Processes reference image for phase correlation with frames.
    ref = rigid.phasecorr_reference(refImg, ops["smooth_sigma"])
    data = rigid.apply_masks(
        zstack.astype(np.float32),
        *rigid.compute_masks(
            refImg=refImg,
            maskSlope=ops["spatial_taper"]
            if ops["1Preg"]
            else 3 * ops["smooth_sigma"],
        )
    )
    # Performs rigid phase correlation between the Z stack and the ref image.
    corrRes = rigid.phasecorr(
        data,
        ref.astype(np.complex64),
        ops["maxregshift"],
        ops["smooth_sigma_time"],
    )
    # Gets the maximum shifts in x and y.
    maxCor = np.argmax(corrRes[-1])
    dx = corrRes[1][maxCor]
    dy = corrRes[0][maxCor]
    zstackCorrected = np.zeros_like(zstack)

    # Shifts every plane according to the phase corr with the ref image.
    for z in range(zstack.shape[0]):
        frame = zstack[z, :, :]
        zstackCorrected[z, :, :] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
    return zstackCorrected


def register_zstack(
    tiff_path, spacing=1, piezo=None, target_image=None, channel=1
):
    """
    Loads tiff file containing imaged z-stack, aligns all frames to each other,
    averages across repetitions, and (if piezo not None) reslices the 3D
    z-stack so that slant/orientation of the new slices matches the slant of
    the frames imaged during experiments (slant given by piezo trace).

    Parameters
    ----------
    tiff_path : String
        Path to tiff file containing z-stack. Note the assumed format of the
        z stack is [planes,frames,X,Y] with frames referring to the snapshots
        taken at one plane.
    spacing: int
        distance between planes of the Z stack (in microns).
    piezo : np.array [t]
        Movement of piezo across z-axis for one plane. Unit: microns. Raw taken
        from niDaq. [Note: need to add more input arguments depending on how
        registration works. Piezo movement might need to provided in units of
        z-stack slices if tiff header does not contain information about depth
        in microns]
    target_image : np.array [x x y]
        Image used by suite2p to align frames to. Is needed to align z-stack
        to this image and then apply masks at correct positions.

    Returns
    -------
    zstack : np.array [x x y x z]
        Registered (and resliced) z-stack.
    """
    # Loads Z stack.
    image = skimage.io.imread(tiff_path)
    # there are two channel
    if image.ndim > 4:
        image = image[:, :, channel - 1, :, :]

    # Gets the number of planes and no. of pixels along X and Y.
    planes = image.shape[0]
    resolutionx = image.shape[2]
    resolutiony = image.shape[3]
    # Prepares an array where the processed Z stack planes will be placed.
    zstack = np.zeros((planes, resolutionx, resolutiony))

    for i in range(planes):
        # Uses the suite2p registration function to align the 10 frames taken
        # per plane to the first frame in each plane.
        res = register_frames(
            image[i, 0, :, :], image[i, :, :, :].astype(np.int16), bidiphase=1
        )

        # Calculates the mean of those 10 registered frames per plane.
        zstack[i, :, :] = np.mean(res[0], axis=0)
    # Performs local registration of the Z stack using the neighboring planes
    # as reference.
    zstack = register_zstack_frames(zstack)

    # Unless there is no piezo trace, the Z stack is slanted according to the
    # piezo movement. The frames are acquired using fast imaging
    # (sawtooth) which means that along the y axis the Z differs. This is
    # different to taking the Z stack which uses slow imaging.
    if not (piezo is None):
        # Normalises the piezo depending on the spacing between planes.
        piezoNorm = piezo / spacing

        zstackTmp = np.zeros(zstack.shape)

        # Changes the slant of each plane of the Z stack using the function
        # _fill_plane_piezo. See function for details.
        for p in range(planes):
            zstackTmp[p, :, :] = _fill_plane_piezo(zstack, piezoNorm, p)
        # apply a gaussian filter of 1 sigma on the y axis
        zstack = zstackTmp

    if not (target_image is None):
        # Registers the z Stack to the reference image using functions from
        # suite2p. See function for details.
        zstack = register_stack_to_ref(zstack, target_image)

    # zstack = sp.ndimage.gaussian_filter(zstack, (0, 1, 0))
    return zstack


# TODO


def extract_zprofiles(
    extraction_path,
    zstack,
    neuropil_correction=None,
    ROI_masks=None,
    neuropil_masks=None,
    smoothing_factor=None,
    metadata={},
    abs_zero=None,
):
    """
    Extracts fluorescence of ROIs across depth of z-stack.

    Parameters
    ----------
    extraction_path: str
        The current directory path.
    zstack : np.array [x x y x z]
        Registered z-stack where slices are oriented the same way as imaged
        planes (output of register_zstack).
    neuropil_correction : np.array [nROIs]
        Correction factors determined by preprocess_traces.correct_neuropil.
    ROI_masks : np.array [x x y x nROIs]
        (output of suite2p so need to check the format of their ROI masks)
        Pixel masks of ROIs in space (x- and y-axis).
    neuropil_masks : np.array [x x y x nROIs]
        (this assumes that suite2p actually uses masks for neuropil)
        Pixel masks of ROI's neuropil in space (x- and y-axis).
    smoothing_factor:



    Returns
    -------
    zprofiles : np.array [z x nROIs]
        Depth profiles of ROIs.
    """
    """
    Steps
    1) Extracts fluorescence within ROI masks across all slices of z-stack.
    2) Extracts fluorescence within neuropil masks across all slices of z-stack.
    3) Performs neuropil correction on ROI traces using neuropil traces and 
    correction factors.
    4) Smoothes the Z profile traces with a gaussian filter.
    
    Notes (useful functions in suite2p);
    - neuropil masks are created in 
    /suite2p/extraction/masks.create_neuropil_masks called from 
    masks.create_masks
    - ROI and neuropil traces extracted in 
    /suite2p/extraction/extract.extract_traces called from 
      extract.extraction_wrapper
    - to register frames, see line 285 (rigid registration) in 
    /suite2p/registration/register for rigid registration
    """
    # Loads suite2p outputs stat, ops and iscell.
    stat = np.load(
        os.path.join(extraction_path, "stat.npy"), allow_pickle=True
    )
    ops = np.load(
        os.path.join(extraction_path, "ops.npy"), allow_pickle=True
    ).item()
    isCell = np.load(os.path.join(extraction_path, "iscell.npy")).astype(bool)

    # Gets the resolution in X and Y of the z stack.
    X = zstack.shape[1]
    Y = zstack.shape[2]

    if (ROI_masks is None) and (neuropil_masks is None):
        # Suite2P function: creates cell and neuropil masks.
        rois, npils = create_masks(stat, Y, X, ops)

    # Gets the "fluorescence traces" for each ROI within the Z stack. Treats
    # each plane in the Z stack like a frame in time; this is the same function
    # that is used to extract the F and N traces.
    # Aditionally extracts the neuropil traces.
    zProfile, Fneu = extract_traces(zstack, rois, npils, 1)

    # Adds the zero signal value. Refer to function for further details.
    if (abs_zero is None):
        zProfile = zero_signal(zProfile)
        Fneu = zero_signal(Fneu)
    else:
        zProfile = zero_signal(zProfile, abs_zero)
        Fneu = zero_signal(Fneu, abs_zero)

    # Only takes the ROIs which are considered cells.
    zProfile = zProfile[isCell[:, 0], :].T
    Fneu = Fneu[isCell[:, 0], :].T

    neuMin = np.nanmin(Fneu, 0)
    ZprofMin = np.nanmin(zProfile, 0)

    FneuRaw = Fneu.copy()
    zprofileRaw = zProfile.T.copy()

    Fneu -= neuMin
    zProfile -= ZprofMin
    # Performs neuropil correction of the zProfile.
    if not (neuropil_correction is None):
        zProfile = np.fmax(zProfile - (neuropil_correction[1, :].reshape(
            1, -1) * Fneu + neuropil_correction[0, :].reshape(1, -1)), 0)
        # iF - (b * iN + a) + F0[:, iROI]
    #
    zProfile += ZprofMin
    # Smoothes the Z profile using a gaussian filter.
    if not (smoothing_factor is None):
        zProfile = sp.ndimage.gaussian_filter1d(
            zProfile, smoothing_factor, axis=0
        )

    # Appends the raw and neuropil corrected Z profiles into a dictionary.
    metadata["zprofiles_raw"] = zprofileRaw
    metadata["zprofiles_neuropil"] = Fneu.T

    return zProfile
