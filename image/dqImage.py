#! /usr/bin/env python
from __future__ import print_function, division
from astropy.io import fits
import matplotlib.pyplot as plt

"""flag the bad pixel
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


def dqImage(fn, flagList=[3, 16, 32, 256]):
    dq = fits.getdata(fn, 3)
    obsType = fits.getval(fn, 'filter', ext=0)
    if type(flagList) is not (list or tuple):
        flag = flagList
        fig, ax = plt.subplots()
        dqImage = (dq // flag) % 2
        ax.matshow(dqImage, origin='lower',
                   cmap='gray', label='dq={0:d}'.format(2**flag))
        ax.set_title('{2}, Flag: {0:d}, {1}'.format(
            flag, flag_dict[flag], obsType))
        return fig
    else:
        figList = []
        for i, flag in enumerate(flagList):
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            dqImage = (dq // flag) % 2
            ax.matshow(dqImage, origin='lower',
                       label='dq={0:d}'.format(2**flag), cmap='gray')
            ax.set_title('{2}, Flag: {0:d}, {1}'.format(
                flag, flag_dict[flag], obsType))
            figList.append(fig)
        return figList

if __name__ == '__main__':
    fn = '/Users/ZhouYf/Documents/HST14241/' + \
        'HNPEGB-LO-T/alldata/icytq0e8q_flt.fits'
    figList = dqImage(fn, [256, 8192])
