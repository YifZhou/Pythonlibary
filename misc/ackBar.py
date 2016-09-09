# ! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import itertools
"""ramp effect model

author: Daniel Apai

Version 0.2: add extra keyword parameter to indicate scan or staring
mode observations for staring mode, the detector receive flux in the
same rate during overhead time as that during exposure
precise mathematics forms are included

Version 0.1: Adapted original IDL code to python by Yifan Zhou

"""


def ackBar(
        nTrap,
        eta_trap,
        tau_trap,
        tExp,
        cRates,
        exptime=180,
        trap_pop=0,
        dTrap=[0],
        lost=0,
        mode='scanning'
):
    """Hubble Space Telescope ramp effet model

    Parameters:

    nTrap -- Number of traps in one pixel
    eta_trap -- Trapping efficiency
    tau_trap -- Trap life time
    tExp -- start time of every exposures
    cRate -- intrinsic count rate of each exposures
    expTime -- (default 180 seconds) exposure time of the time series
    trap_pop -- (default 0) number of occupied traps at the beginning of the observations
    dTrap -- (default [0])number of extra trap added in the gap between two orbits
    lost -- (default 0, no lost) proportion of trapped electrons that are not eventually detected
    (mode) -- (default scanning, scanning or staring), for scanning mode
    observation , the pixel no longer receive photons during the overhead
    time, in staring mode, the pixel keps receiving elctrons
    """
    dTrap = itertools.cycle(dTrap)
    obsCounts = np.zeros(len(tExp))
    nTrap = abs(nTrap)
    eta_trap = abs(eta_trap)
    tau_trap = abs(tau_trap)
    for i in xrange(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1 = eta_trap * f_i / nTrap + 1 / tau_trap  # a key factor
        # number of trapped electron during one exposure
        dE1 = (eta_trap * f_i / c1 - trap_pop)[1 - np.exp(-c1 * exptime)]
        trap_pop = trap_pop + dE1
        obsCounts[i] = f_i * exptime - dE1
        if dt < 1200:  # whether next exposure is in next orbits
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between orbits
                dE2 = - trap_pop * (1 - np.exp(-(dt - exptime)/tau_trap))
            else:
                # else there is incoming flux
                dE2 = (eta_trap * f_i / c1 - trap_pop)[1 - np.exp(-c1 * (dt - exptime))]
            trap_pop = min(trap_pop + dE2, nTrap)
        else:
            # next orbits
            trap_pop = min(trap_pop * np.exp(-(dt-exptime)/tau_trap) + next(dTrap), nTrap)
        trap_pop = max(trap_pop, 0)
        # out_trap = max(-(trap_pop * (1 - np.exp(exptime / tau_trap))), 0)
        # out_trap = trap_pop / tau_trap * dt * np.exp(-dt / tau_trap)

    return obsCounts


if __name__ == '__main__':
    t1 = np.linspace(0, 2700, 80)
    t2 = np.linspace(5558, 8280, 80)
    t = np.concatenate((t1, t2))
    crate = 100
    crates = crate * np.ones(len(t))
    dataDIR = '/Users/ZhouYf/Documents/HST14241/alldata/2M0335/DATA/'
    from os import path
    from pmExtractor import pmExtractor
    import pandas as pd

    info = pd.read_csv(
        '/Users/ZhouYf/Documents/HST14241/alldata/2M0335/2M0335_fileInfo.csv',
        parse_dates='Datetime',
        index_col='Datetime')
    info['Time'] = np.float32(info.index - info.index.values[0]) / 1e9
    expTime = info['Exp Time'].values[0]
    grismInfo = info[info['Filter'] == 'G141']
    fnList = [path.join(dataDIR, fn) for fn in grismInfo['File Name']]
    LC, Npix = pmExtractor([60, 200],
                           [148, 149, 150, 151, 152, 153, 154], fnList)
    tExp = grismInfo['Time'].values
    cRates = np.ones(len(LC)) * LC.mean() * 1.002
    obs = ackBar(500, 0.02, 3 * 3600, tExp, cRates, exptime=expTime, lost=0,
                 dTrap=2000)
    plt.close('all')
    plt.plot(tExp, LC*expTime, 'o')
    plt.plot(tExp, obs, '-')
    # plt.ylim([crate * 0.95, crate * 1.02])
    plt.show()
