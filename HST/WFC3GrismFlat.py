#! /usr/bin/env python
"""calcualte the flat field for grism observations
"""
from __future__ import division, absolute_import
from __future__ import print_function
from astropy.io import fits
import numpy as np


def xtowl(x):
    """
    convert the pixel index to wavelength
    """
    return (x * 46.7 + 8953.)


def getFlatCube(arraySize):
    """read flat cube from fits file
    """
    flatCube = np.zeros([arraySize, arraySize, 4])
    startPixel = (1014 - arraySize) / 2
    for i in xrange(4):
        flatCube[:, :, i] = fits.getdata('WFC3.IR.G141.flat.2.fits', i)[startPixel:-startPixel, startPixel:-startPixel]
    return flatCube


def calFlat(x0, arraySize=256):
    """calcualte the wavelength dependent flat field
    """
    flatCube = getFlatCube(arraySize)
    lambda_min = 10600
    lambda_max = 17000
    lambdaList = xtowl(np.arange(arraySize) - x0)
    flat = flatCube[:, :, 0].copy()
    l = (lambdaList - lambda_min) / (lambda_max - lambda_min)
    for i in xrange(arraySize):
        flat[:, i] = flat[:, i] + flatCube[:, i, 1] * l +\
                     flatCube[:, i, 2]*l**2 + flatCube[:, i, 3]*l**3
    return flat

# x0List = [19.915, 23.95, 20.06, 20.16, 20.06, 21.91, 21.94, 23.26, 23.94,
#           21.73, 21.73, 21.73, 21.16, 24.11,
#           24.16]  # x coord of the direct image

if __name__ == '__main__':
    x0 = 20
    flat = calFlat(x0)
    from plot import imshow
    imshow(flat)
