#! /usr/bin/env python
"""bin numpy array to samll size
"""
from __future__ import division, absolute_import
from __future__ import print_function
import numpy as np

def rebin(a, binSize=1, method='mean'):
    """bin the input array a to the input binSize
    Keyword Arguments:
    a       -- input array
    binSize -- (default 1)  aimed bin size
    method  -- combining method, either mean, median, or meansq
    """
    a = np.array(a)
    if a.size % binSize != 0:
        raise Exception('Wrong binSize!\n '
                        'The size of the array needs to be integer times of the binSize')
    else:
        outSize = a.size // binSize
    if method == 'mean':
        # use nanfunction to get a good handle on bad pixels
        return np.nanmean(a.reshape((outSize, binSize)), axis=1)
    elif method == 'median':
        return np.nanmedian(a.reshape((outSize, binSize)), axis=1)
    elif method == 'meansq':
        return np.sqrt(np.nanmean(a.reshape(outSize, binSize)**2, axis=1))
