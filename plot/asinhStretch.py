#!/usr/bin/env python
import numpy as np
from astropy.visualization.stretch import BaseStretch

class asinhStretch(BaseStretch):
    r"""
    An asinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm asinh}(x / a)}{{\rm asinh}(1 / a)}.
    """

    def __init__(self, a=0.1):
        super(asinhStretch, self).__init__()
        self.a = a

    def __call__(self, values, out=None, clip=True):

        values = _prepare(values, out=out, clip=clip)

        np.true_divide(values, self.a, out=values)
        np.arcsinh(values, out=values)
        np.true_divide(values, np.arcsinh(1. / self.a), out=values)

        return values

    @property
    def inverse(self):
        return SinhStretch(a=1. / np.arcsinh(1. / self.a))
