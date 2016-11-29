from __future__ import division
import numpy as np
from scipy import ndimage
from inpaint import replace_nans
import matplotlib.pyplot as plt


def inpaint_array(inputArray, mask):
    # maskedImg = np.ma.array(inputArray, mask=mask)
    # NANMask = maskedImg.filled(np.NaN)
    # badArrays, num_badArrays = ndimage.label(mask)
    # data_slices = ndimage.find_objects(badArrays)
    filled = replace_nans(
        inputArray * mask,
        max_iter=20,
        tol=0.05,
        kernel_radius=5,
        kernel_sigma=2,
        method='idw')
    return filled

if __name__ == '__main__':
    from astropy.io import fits
    from plot import imshow
    plt.close('all')
    im = fits.getdata('WFC3.IR.G141.flat.2.fits')[379:-379, 379:-379]
    dq = np.ones_like(im).astype(float)
    im[50, 100] = 10000
    dq[50, 100] = np.nan
    im[20:24, 120] = 5000
    dq[20:24, 120] = np.nan
    dq[120:128, 100:102] = np.nan
    imshow(im*dq)
    imshow(inpaint_array(im, dq))
    plt.show()
