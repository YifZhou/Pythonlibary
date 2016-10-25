#! /usr/bin/env python
from __future__ import print_function, division
from os import path
import numpy as np
from astropy.io import fits
"""create a mask for sky subtraction
"""


def dqFilter(dq, flagList=[4, 8, 16, 32, 256, 512]):
    DF = np.ones(dq.shape)
    for flag in flagList:
        DF[(dq // flag) % 2 == 1] = np.nan
    return DF


def medianImage(imageCube, axis=2):
    """calculate the median image for sky region construction"""
    return np.nanmedian(imageCube, axis=axis)


def readImageCube(FNList, dataDIR, dq, nSamp=1, imageSize=256):
    """read in the image cube, and map bad data with flags"""
    imageCube = np.zeros((imageSize, imageSize, len(FNList)))
    for i, fn in enumerate(FNList):
        im = fits.getdata(path.join(dataDIR, fn), ('sci', nSamp))[5:261, 5:261]
        imageCube[:, :, i] = im * dq
    return imageCube


def makeMask(mask0, masks, saveFN):
    """create mask to maskout the region with scanning spectra"""
    for mask in masks:
        mask0[mask[1]-5:mask[3]+5, mask[0]-5:mask[2]+5] = np.nan
    fits.writeto(saveFN, mask0, clobber=True)
    return mask0


if __name__ == '__main__':
    masks = [
        [[60, 150, 188, 235], [100, 39, 224, 120], [76, 50, 199, 136]],
        [[66, 144, 190, 239], [68, 26, 193, 128], [50, 49, 174, 136]],
        [[60, 152, 185, 234], [106, 38, 229, 124], [82, 52, 204, 135]],
        [[60, 150, 185, 224], [117, 42, 236, 129], [91, 55, 220, 140]],
        [[60, 150, 185, 235], [117, 44, 243, 129], [90, 56, 218, 139]],
        [[65, 144, 186, 242]], [[66, 143, 183, 240]], [[66, 154, 189, 234]],
        [[65, 142, 190, 242], [56, 28, 179, 124]], [[64, 144, 194, 239]],
        [[65, 142, 193, 239]], [[66, 144, 199, 240]], [[64, 144, 189, 237]],
        [[65, 144, 189, 242], [39, 34, 161, 129]],
        [[69, 142, 190, 240], [66, 30, 187, 126]]
    ]  # mask for 15 visits
    dqMask = dqFilter(fits.getdata('commonDQ.fits', 0))
    for visit in xrange(15):
        saveFN = 'skyMask_visit_{0:02d}.fits'.format(visit+1)
        mask0 = dqMask.copy()
        makeMask(mask0, masks[visit], saveFN)
