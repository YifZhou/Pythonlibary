#!/usr/bin/env python
"""Gaussian function
"""
import numpy as np

def gaussian(x, mu=0, sigma=1.0):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2 / (2 * sigma**2))
