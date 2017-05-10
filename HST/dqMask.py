#! /usr/bin/env python

import numpy as np

u"""flag the bad pixel and create mask for the bad pixels
"""

from __future__ import absolute_import
flag_dict = {
    0: u'OK',
    1: u'decoding error',
    2: u'data missing',
    4: u'bad pixel',
    8: u'non-zero bias',
    16: u'hot pixel',
    32: u'unstable response',
    64: u'warm pixel',
    128: u'bad reference',
    256: u'saturation',
    512: u'bad flat',
    2048: u'signal in zero read',
    4096: u'CR by MD',
    8192: u'cosmic ray',
    16384: u'ghost'
}


def dqMask(dq, flagList=[4, 16, 32, 256]):
    u"""identify certain flagged pixels as bad pixels, and create mask
    Return
      mask -- bool array, masked pixel marked as True
    Parameters:
      fnList -- file List to get the dq image
      imageDim -- (default 256 subframe) dq image size
      flagList -- flag used to identify bad pixels
    """
    dqMask = np.zeros_like(dq, dtype=float)
    for flag in flagList:
        dqMask += dq // flag % 2
    dqMask = dqMask.astype(bool)
    return dqMask


if __name__ == u'__main__':
    pass
