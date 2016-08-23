# ! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import itertools
"""ramp effect model
author: Daniel Apai
Adapted to python by Yifan Zhou
"""


def ackBar(
        nTrap,  # number of electron traps
        eta_trap,  # trapping efficiency
        tau_trap,  # trapping time scale
        tExp,  # start of the every exposure, in second
        cRates,  # count Rate
        exptime=180,  # exposure time
        trap_pop=0,
        dTrap=[0],  # extra trapped electron added between orbits
        lost=0  # whether the trapped electron is lost or detected
):
    """return count"""
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
        effective_cRate = cRates[i]
        incoming = exptime * effective_cRate
        in_trap1 = (nTrap - trap_pop) * (1 - np.exp(-eta_trap*incoming/nTrap))
        # when eta*incoming << nTrap, this is similar to previous equation
        # possible to be dectected
        out_trap1 = trap_pop * (1 - np.exp(-exptime/tau_trap))
        obsCounts[i] = (
            incoming - in_trap1 + (1 - lost) * out_trap1)
        # electron released before next exposure
        in_trap2 = (nTrap - trap_pop) *\
                   (1 - np.exp(-eta_trap*dt*effective_cRate/nTrap))
        out_trap2 = trap_pop * (1 - np.exp(-dt/tau_trap))
        if dt > 1200:  # whether next exposure is in next orbits
            # using in_trap1, because god know what happened to the detector
            # between orbits
            trap_pop = min(trap_pop + in_trap1 - out_trap2 + next(dTrap), nTrap)
        else:
            trap_pop = min(trap_pop + in_trap2 - out_trap2, nTrap)
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
