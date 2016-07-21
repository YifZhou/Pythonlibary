#! /usr/bin/env python
"""make median combined images from a list of image
usually for make image template
"""
from __future__ import division, absolute_import
from __future__ import print_function
from astropy.io import fits
import numpy as np

def medianImage(fnList, imageDim=256, dq=None):
    """make the median combined images

    Keyword Arguments:
    fnList -- list of input file names
    imageDim -- (default 256 subframe) input image size
    dq -- (default is None) dataQuality Array, if dq is None, do not
    do any data quality examination for the tempalte
    """
    imageCube = np.zeros((imageDim, imageDim, len(fnList)))
    for i, fn in enumerate(fnList):
        imageCube[:, :, i] = fits.getdata(fn, 1)
    median = np.median(imageCube, axis=2)
    if dq is None:
        return median
    else:
        return median * dq
