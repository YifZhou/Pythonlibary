#!/usr/bin/env python
"""2D gaussian profile"""
import numpy as np

def gaussian2d(x, y, amp, x0, y0, sigma_x, sigma_y):
    return amp * np.exp(-(x-x0)**2 / (2*sigma_x**2)
                        -(y-y0)**2 / (2*sigma_y**2))
