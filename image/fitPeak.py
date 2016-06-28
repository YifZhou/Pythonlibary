#!/usr/bin/env python
from functions import gaussian2d
from lmfit import Model
import numpy as np
"""use Gaussian profile to fit profile peak"""


def fitPeak(im,
            x0,
            y0,
            subSize=10,
            init_params={'amp': 10,
                         'x0': 10,
                         'y0': 10,
                         'sigma_x': 1.0,
                         'sigma_y': 1.0}):
    subImage = im[y0 - subSize:y0 + subSize, x0 - subSize:x0 + subSize]
    peakModel = Model(gaussian2d, independent_vars=['x', 'y'])
    x, y = np.meshgrid(range(2 * subSize), range(2 * subSize))
    p = peakModel.make_params(amp=init_params['amp'],
                              x0=init_params['x0'],
                              y0=init_params['y0'],
                              sigma_x=init_params['sigma_x'],
                              sigma_y=init_params['sigma_y'])
    result = peakModel.fit(subImage,
                           x=x,
                           y=y,
                           params=p, method='powell')
    return (result.values['x0'] + x0 - subSize,
            result.values['y0'] + y0 - subSize)
