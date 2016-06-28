#!/usr/bin/env python
"""simulate a sin wave observed by HST
default orbit length 96 min,
default visibility per orbit, 50 min
defaut overhead 20s
return the electron count for each exposure
"""

from __future__ import division, absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import batman
from HST.obsTime import obsTime
def HSTtransitLC(expTime, cRate,
                 param,
                 orbits=4,
                 orbitLength=96,  # min
                 visibility=50,  # min
                 overhead=20,  # s
                 plot=False):
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = param['t0'] / (24 * 60)  # time of inferior conjunction
    params.per = param['period']  # orbital period
    params.rp = param['rp']  # planet radius (in units of stellar radii)
    params.a = 15.23  # semi-major axis (in units of stellar radii)
    params.ecc = 0  # eccentricity
    params.inc = 89.1  # inclination
    params.w = 90.  # longitude of periastron (in degrees)
    params.limb_dark = "linear"  # limb darkening model
    params.u = [0.28]  # limb darkening coefficients

    t = obsTime(expTime, orbits, orbitLength, visibility, overhead)
    m = batman.TransitModel(params, t / (24 * 3600))
    # calculate count, be careful that period in h
    count = cRate * expTime * m.light_curve(params)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(t / 60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('count [$\mathsf{e^-}$]')
    return count, t


if __name__ == '__main__':
    plt.close('all')
    # for GJ 1214b # from Kreidberg 2013
    params = {}
    params['period'] = 1.58040464894  # days
    params['rp'] = 0.0134**0.5  # stellar radii
    params['t0'] = 220
    count, t = HSTtransitLC(88.9, 250, params, orbits=4, plot=True, overhead=50)
    plt.show()
