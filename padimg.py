#! /usr/bin/env python

"""
substitute IDL version of padimg, try to mimic the behavior of the IDL code exactly
"""
from __future__ import print_function, division
import glob
from astropy.io import fits
from HST import dqMask
import numpy as np
from scipy.interpolate import griddata
from os.path import basename


def interpBadPixels(image, mask):
    """Use simple linear interpolation to fix bad pixel

    :param image: original image
    :param mask: mask in which bad pixels are marked as >=1

    """
    goody, goodx = np.where(mask == 0)
    bady, badx = np.where(mask > 0)
    fixedIm = image.copy()
    interpv = griddata(
            (goody, goodx),
            image,
            (bady, badx),
            method='linear')
    fixedIm[bady, badx] = interpv
    return fixedIm


def padimg(fname, zero=True, correct=True, skyname=None, flatname=None):
    """Micmic the IDL code to pad images for aXe use

    :param fname: name patterns
    :param zero: (default True) if zero is True, use zero to pad the image, else use nan
    :param correct: (default True) if correct is True, correct bad pixels
    :param skyname: sky background for correction
    :param flatname: flat field files for correction

    """
    FNs = glob.glob(fname)
    FNs.sort()
    # make corrections for sky and flat first
    FN0 = FNs[0]
    dq0 = fits.getdata(FN0, 'dq')
    mask0 = dqMask(dq0)
    subarray = dq0.shape[0]  # size of the subarray
    fullarray = 1014  # size of WFC3 full array
    index0 = (fullarray - subarray) // 2  # start index of subarray
    # interpolate sky
    if skyname is not None:
        print('Processing sky background file: {0}'.format(basename(skyname)))
        with fits.open(skyname) as skyf:
            # take subarray
            sky0 = skyf[0].data[index0:index0+subarray, index0:index0+subarray]
            if correct:
                sky0 = interpBadPixels(sky0, mask0)
            skyf[0].data[index0:index0+subarray, index0:index0+subarray] = sky0
            skyf.writeto(skyname.replace('.fits', '_xp.fits'))

    # interpolate flat field
    if flatname is not None:
        print('Processing flat field file: {0}'.format(basename(flatname)))
        with fits.open(flatname) as flatf:
            for i in range(4):  # flat field is 3 order polynomial
                flat_i = flatf[i].data[index0:index0+subarray, index0:index0+subarray]
                if correct:
                    flat_i = interpBadPixels(flat_i, mask0)
                flatf[i].data[index0:index0+subarray, index0:index0+subarray] = flat_i
            flatf.writeto(flatname.replace('.fits', '_xp.fits'))

    for FN in FNs:
        print('Processing data file: {0}'.format(basename(FN)))
        with fits.open(FN) as f:
            # Extension 0
            f[0].header['SUBARRAY'] = False
            # EXtension 1
            subImage = f[1].data
            if zero:
                fullImage = np.full((fullarray, fullarray), 0, dtype='float')
            else:
                fullImage = np.full((fullarray, fullarray), np.nan, dtype='float')
            if correct:
                subImage = interpBadPixels(subImage, mask0)
            fullImage[index0:index0+subarray, index0:index0+subarray] = subImage
            f[1].data = fullImage
            # Extension 2
            fullError = np.zeros_like(fullImage)
            fullError[index0:index0+subarray, index0:index0+subarray] = f[2].data
            f[2].data = fullError
            # Extension 3
            fullDQ = np.full((fullarray, fullarray), 4, dtyp='int')
            fullDQ[index0:index0+subarray, index0:index0+subarray] = f[3].data
            f[3].data = fullDQ
            # Extension 4
            fullTime = np.zeros_like(fullImage)
            fullTime[index0:index0+subarray, index0:index0+subarray] = f[4].data
            f[4].data = fullTime
            # write file
            f.writeto(FN.replace('.fits', '_xp.fits'))
