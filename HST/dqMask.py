#! /usr/bin/env python
from __future__ import print_function, division
from astropy.io import fits
import numpy as np

"""flag the bad pixel and create mask for the bad pixels
"""

flag_dict = {
    0: 'OK',
    1: 'decoding error',
    2: 'data missing',
    4: 'bad pixel',
    8: 'non-zero bias',
    16: 'hot pixel',
    32: 'unstable response',
    64: 'warm pixel',
    128: 'bad reference',
    256: 'saturation',
    512: 'bad flat',
    2048: 'signal in zero read',
    4096: 'CR by MD',
    8192: 'cosmic ray',
    16384: 'ghost'
}


def dqMask(fnList, imageDim=256, flagList=[4, 16, 32, 256]):
    """identify certain flagged pixels as bad pixels
    mark the bad pixels as nan

    Parameters:
      fnList -- file List to get the dq image
      imageDim -- (default 256 subframe) dq image size
      flagList -- flag used to identify bad pixels
    """
    dqMask = np.zeros((imageDim, imageDim))
    for i, fn in enumerate(fnList):
        dq = fits.getdata(fn, 'dq')
        for flag in flagList:
            dqMask += dq // flag % 2
    dqMask[dqMask != 0] = np.nan
    dqMask[~np.isnan(dqMask)] = 1
    return dqMask

if __name__ == '__main__':
    pass
