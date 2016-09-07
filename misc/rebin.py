#! /usr/bin/env python
"""bin numpy array to samll size
"""
from __future__ import division, absolute_import
from __future__ import print_function
import numpy as np


def rebin(a, binSize=1):
    """bin the input array a to the input binSize
    Keyword Arguments:
    a       -- input array
    binSize -- (default 1)  aimed bin size
    """
    if a.size() % binSize != 0:
        raise Exception('Wrong binSize!\n '
                        'The size of the array needs to be integer times of the binSize')
    else:
        outSize = a.size() // binSize
    return a.reshape((outSize, binSize)).mean(axis=1)
