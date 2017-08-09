#! /usr/bin/env python
"""calcualte the flat field for grism observations
"""


from astropy.io import fits
import numpy as np


def wlDispersion(xc, yc):
    """
    convert the pixel index to wavelength
    """
    DLDP0 = [8949.40742544, 0.08044032819916265]
    DLDP1 = [44.97227893276267,
             0.0004927891511929662,
             0.0035782416625653765,
             -9.175233345083485e-7,
             2.2355060371418054e-7,  -9.258690000316504e-7]
    # calculate field dependent dispersion coefficient
    p0 = DLDP0[0] + DLDP0[1] * xc
    p1 = DLDP1[0] + DLDP1[1] * xc + DLDP1[2] * yc + \
         DLDP1[3] * xc**2 + DLDP1[4] * xc * yc + DLDP1[5] * yc**2
    dx = np.arange(1014) - xc
    return (p0 + dx * p1)


def calFlat(xc, yc, outSize=256):
    """
    x0 the x corrdinate of the source, in detector coordinate
    """
    # read in the flat cube
    flatCube = np.zeros((1014, 1014, 4))
    for i in range(4):
        flatCube[:, :, i] = fits.getdata('WFC3.IR.G141.flat.2.fits', i)
    lambda_min = 10600
    lambda_max = 17000
    lambdaList = wlDispersion(xc, yc)
    flat = flatCube[:, :, 0].copy()
    l = (lambdaList - lambda_min) / (lambda_max - lambda_min)
    for i in range(1014):
        flat[:, i] = flat[:, i] + flatCube[:, i, 1] * l +\
                     flatCube[:, i, 2]*l**2 + flatCube[:, i, 3]*l**3
    margin = (1014 - outSize) // 2
    return flat[margin:-margin, margin:-margin]

# x0List = [19.915, 23.95, 20.06, 20.16, 20.06, 21.91, 21.94, 23.26, 23.94,
#           21.73, 21.73, 21.73, 21.16, 24.11,
#           24.16]  # x coord of the direct image


# flatCube = np.zeros([256, 256, 4])
