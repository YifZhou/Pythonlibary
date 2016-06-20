#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy import visualization
from astropy.visualization.mpl_normalize import ImageNormalize

__all__ = ['imshow']
def imshow(im,
           vmin=None,
           vmax=None,
           stretch='asinh',
           clip=False,
           ax=None,
           cmap='viridis',
           interpolation='nearest',
           plotColorbar=True,
           percentile=0.995):
    """priorily using asinh to strech the image
    """
    if vmin is None:
        vmin = np.percentile(im, ((1 - percentile) / 2) * 100)
    if vmax is None:
        vmax = np.percentile(im, ((1 - percentile) / 2 + percentile) * 100)
    if stretch == 'asinh':
        stretcher = visualization.AsinhStretch()
    elif stretch == 'log':
        stretcher = visualization.LogStretch()
    else:
        # if stretch method is not specified, use linear
        stretcher = visualization.LinearStretch()
    normalizer = ImageNormalize(vmin=vmin,
                                vmax=vmax,
                                stretch=stretcher,
                                clip=clip)

    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.matshow(im,
                     norm=normalizer,
                     origin='lower',
                     cmap=cmap,
                     interpolation=interpolation)
    if plotColorbar:
        fig.colorbar(cax, spacing='uniform')
    return cax


if __name__ == '__main__':
    from astropy.io import fits
    im = fits.getdata('ibxy01akq_flt.fits', 1)
    cax = imshow(im, vmin=-50, vmax=50, stretch='linear')
    plt.show()
