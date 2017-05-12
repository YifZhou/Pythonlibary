#! /usr/bin/env python

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


def dqMask(dq, flagList=[4, 16, 32, 256]):
    """identify certain flagged pixels as bad pixels, and create mask
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


if __name__ == '__main__':
    pass
