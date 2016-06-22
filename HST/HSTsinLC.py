#!/usr/bin/env python
"""simulate a sin wave observed by HST
default orbit length 96 min,
default visibility per orbit, 50 min
defaut overhead 20s
return the electron count for each exposure
"""
from __future__ import division, absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from obsTime import obsTime
def HSTsinLC(expTime, cRate, param,
             orbits=4,
             orbitLength=96,  # min
             visibility=50,  # min
             overhead=20,  # s
             plot=False):
    t = obsTime(expTime, orbits, orbitLength, visibility, overhead)
    # calculate count, be careful that period in h
    count = cRate * expTime * \
        (1 + np.sin((2 * np.pi * t / (param['period'] * 3600)) +
                    param['phase']) * param['amplitude'])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(t/60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('count [$\mathsf{e^-}$]')
    return count, t


if __name__ == '__main__':
    plt.close('all')
    param = {}
    param['period'] = 14.3
    param['phase'] = np.random.uniform(0, 2*np.pi, 1)
    param['amplitude'] = 0.05
    count, t = HSTsinLC(88.9, 250, param,
                        orbits=6, plot=True)
    plt.show()
