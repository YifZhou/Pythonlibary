#! /usr/bin/env python
u"""interpolate bad pixels using median of surrounding pixels
"""
from __future__ import absolute_import
import numpy as np
from scipy.interpolate import griddata
from itertools import izip
# from scipy.interpolate import SmoothBivariateSpline


def badPixelInterp(im, mask):
    u"""im -- image to be interpolated
       mask -- bad pixel mask, a boolean array, mark bad pixel as True
    """
    return_im = im.copy()
    bad_i, bad_j = np.where(mask)  # identify bad pixels
    for i, j in izip(bad_i, bad_j):
        # loop over different pixels
        i_low = max(i - 4, 0)
        i_high = i + 4
        j_low = max(j - 4, 0)
        j_high = j + 4
        # return_im[i, j] = np.nanmean(im[i_low:i_high, j_low:j_high])
        i_list, j_list = np.where(mask[i_low:i_high, j_low:j_high] == 0)
        try:
            return_im[i, j] = griddata(list(izip(i_list, j_list)),
                                       im[i_low+i_list, j_low+j_list],
                                       (i-i_low, j-j_low),
                                       method=u'linear')
        except Exception, e:
            return_im[i, j] = np.nanmean(im[i_low+i_list, j_low+j_list])
    return return_im


if __name__ == u'__main__':
    u"""test the function"""
    from astropy.io import fits
    from plot import imshow
    import matplotlib.pyplot as plt
    plt.close(u'all')
    im = fits.getdata(u'WFC3.IR.G141.flat.2.fits')[379:-379, 379:-379]
    dq = np.ones_like(im).astype(float)
    im[50, 100] = 10000
    dq[50, 100] = np.nan
    im[20:24, 120] = 5000
    dq[20:24, 120] = np.nan
    imshow(im*dq)
    imshow(badPixelInterp(im, dq))
    plt.show()
