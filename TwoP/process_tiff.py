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
from suite2p.registration.register import register_frames, compute_reference, shift_frames
from suite2p.registration import rigid
from TwoP.preprocess_traces import correct_neuropil
from suite2p.default_ops import default_ops
from numba import jit

from TwoP.preprocess_traces import zero_signal


# @jit(forceobj=True)
def _reslice_zstack(stack, piezo, spacing=1):
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
    piezo : array [t]
        the piezo depth.
    i: int
        the frame to create
    Returns
    -------
    a normalised frame.

    """
    stack_resliced = np.zeros_like(stack)

    # Set top of piezo trace to zero.
    piezo -= piezo[0]
    # Convert piezo values from microns to z stack spacing.
    piezoNorm = piezo / spacing

    # Gets the number of planes and no. of pixels along X and Y of the Z stack.
    planes = stack.shape[0]
    resolutionx = stack.shape[1]
    resolutiony = stack.shape[2]

    # Transfer function: imaged line -> plane in Z stack.
    f = sp.interpolate.interp1d(np.linspace(
        0, resolutiony, len(piezoNorm)), piezoNorm[:, 0])
    piezoNorm = f(np.arange(0, resolutiony))
    # Set centre of piezo trace to zero. Will be the top most plane in the resliced Z stack.
    piezoNorm -= piezoNorm[int(np.round(len(piezoNorm) / 2))]

    xx = np.arange(0, resolutionx)
    yy = np.arange(0, resolutiony)

    # Creates an interpolating function based on the z stack.
    interp = sp.interpolate.RegularGridInterpolator(
        (
            np.arange(0, planes),
            yy,
            xx,
        ),
        stack,
        fill_value=None,
        bounds_error=False,
        method='nearest'
    )

    X, Y = np.meshgrid(xx, yy, indexing='xy')
    Z0 = np.tile(piezoNorm.reshape(-1, 1), (1, resolutionx))

    for p in range(planes):
        Z = Z0 + p
        # Clip values outside the bounds to the bounds of the stack.
        Z = np.clip(Z, 0, (planes - 1) * spacing)
        # Reslices the Z stack plane according to the piezo trace.
        plane_points = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1)
        stack_resliced[p, :, :] = interp(plane_points).reshape(X.shape)

    return stack_resliced


def register_zstack_frames(zstack, ops):
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
    ops : dict
        suite2p settings.

    Returns
    -------
    zstack : np.ndarray [planes, x, y]
        The locally registered Z stack.

    """
    # Calculates shifts in x and y for each plane in the Z stack to align with its previous plane.
    y_off = np.zeros(zstack.shape[0])
    x_off = np.zeros(zstack.shape[0])
    for i in range(zstack.shape[0] - 1):
        registration = register_frames(zstack[i, :, :], np.expand_dims(zstack[i + 1], axis=0).astype(np.float32),
                                       ops=ops)
        y_off[i+1] = registration[1][0]
        x_off[i+1] = registration[2][0]
    # Add up shifts consecutively to align the whole z stack.
    y_off = np.cumsum(y_off, axis=0)
    x_off = np.cumsum(x_off, axis=0)
    # Minimize total shifts by subtracting the median in each direction.
    y_off = y_off - np.median(y_off)
    x_off = x_off - np.median(x_off)
    # Apply the shifts to the z stack.
    zstack = shift_frames(zstack.astype(np.float32), yoff=y_off.astype(int), xoff=x_off.astype(int), yoff1=None, xoff1=None,
                          ops=ops)
    return zstack


def register_stack_to_ref(zstack, refImg, ops=default_ops()):
    """
    Registers the Z stack to the reference image using the same approach
    as registering the frames to the reference image.
    All functions come from suite2p, see their docs for further information.

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
    stack_with_mask = rigid.apply_masks(
        zstack.astype(np.float32),
        *rigid.compute_masks(
            refImg=refImg,
            maskSlope=3 * ops["smooth_sigma"],
        )
    )
    # Performs rigid phase correlation between the Z stack and the ref image.
    corrRes = rigid.phasecorr(
        stack_with_mask,
        ref.astype(np.complex64),
        ops["maxregshift"],
        ops["smooth_sigma_time"],
    )
    # Gets the shifts in x and y for the zstack-plane most correlated with the reference image.
    maxCor = np.argmax(corrRes[-1])
    dx = corrRes[1][maxCor]
    dy = corrRes[0][maxCor]
    zstackCorrected = shift_frames(zstack.astype(np.float32), yoff=np.tile(dy, zstack.shape[0]),
                                   xoff=np.tile(dx, zstack.shape[0]), yoff1=None, xoff1=None, ops=ops)
    return zstackCorrected


def register_zstack(
    tiff_path, ops, spacing=1, piezo=None, target_image=None, channel=1
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
    ops : dict
        suite2p settings.
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
    # If there are two channel, choose input channel.
    if image.ndim > 4:
        image = image[:, :, channel - 1, :, :]

    # Average repeated frames per plane in Z stack.
    planes = image.shape[0]
    repetitions = image.shape[1]
    resolutionx = image.shape[2]
    resolutiony = image.shape[3]
    zstack = np.zeros((planes, resolutionx, resolutiony))
    for i in range(planes):
        # Uses the suite2p registration function to align the repeated frames taken
        # per plane to the middle repetition of each plane.
        res = register_frames(
            image[i, int(np.round(repetitions/2)), :, :], image[i, :, :, :], ops=ops
        )
        # Calculates the mean across repeated frames per plane.
        zstack[i, :, :] = np.mean(res[0], axis=0)

    # Registers planes of Z stack to each other.
    zstack = register_zstack_frames(zstack, ops)

    # Unless there is no piezo trace, the Z stack is resliced according to the
    # piezo movement. The frames are acquired using fast imaging
    # (sawtooth) which means that along the y axis the Z differs. This is
    # different to taking the Z stack which uses slow imaging.
    # TODO (SS): make sure that piezo and target_image are not None.

    # Changes the slant of each plane of the Z stack.
    zstack = _reslice_zstack(zstack, piezo, spacing=spacing)
    # TODO (SS): smooth zstack planes!
    #  And perform non-rigid registration with target_image!
    # Registers the z Stack to the reference image.
    zstack = register_stack_to_ref(zstack, target_image, ops)
    return zstack


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
    zstack : np.array [Z x Y x X]
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
    X = zstack.shape[2]
    Y = zstack.shape[1]

    if (ROI_masks is None) and (neuropil_masks is None):
        # Suite2P function: creates cell and neuropil masks.
        rois, npils = create_masks(stat, Y, X, ops)

    # Gets the "fluorescence traces" for each ROI within the Z stack. Treats
    # each plane in the Z stack like a frame in time; this is the same function
    # that is used to extract the F and N traces.
    # Aditionally extracts the neuropil traces.
    zProfile, Fneu = extract_traces(zstack, rois, npils)

    # Adds the zero signal value. Refer to function for further details.
    if abs_zero is None:
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
