#!/usr/bin/env python
"""generate a series of time for the starting of exposure
default orbit length 96 min,
default visibility per orbit, 50 min
defaut overhead 20s
return the electron count for each exposure
return the series in seconds
"""
import numpy as np

def obsTime(exptime,
         orbits=4,
         orbitLength=96,  # min
         visibility=50,  # min
         overhead=20):
    t = np.arange(0, visibility * 60, (exptime + overhead))
    t0 = np.arange(orbits) * orbitLength * 60  # starting time of each orbits
    return np.concatenate([t + t0i for t0i in t0])
