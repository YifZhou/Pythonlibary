# -*- coding: utf-8 -*-
# /usr/bin/env python
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import corner
import numpy as np
from os import path
"""plot mcmc fit result
"""


def plotMCMC(chain, modelFunc, modelArgs, modelX, obsY,
             argName=None, burnin=100, saveDIR=None):
    """plot MCMC result
    Keyword Arguments:
    chain     -- mcmc result
    modelFunc -- model function
    modelArgs -- model function arguments
    modelX    -- x axis argument for the plot
    obsY      -- observed value
    argName   -- (default None)
    burnin    -- (default 100) burn in steps
    saveDIR   -- (default None)  save directory, if None, don't save
    """
    chainShape = chain.shape
    nDim = chainShape[2]
    chain = chain[:, burnin:, :].reshape((-1, nDim))  # remove the buring in part
    # figure 1 figure plot
    if argName is None:
        # default arg label is empty
        argName = [''] * nDim
    fig1 = corner.corner(chain, labels=argName)

    # figure 2
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    nRandModel = 50  # plot 50 random model
    for i in np.random.randint(chain.shape[0], size=nRandModel):
        ax.plot(modelX, modelFunc(chain[i, :], *modelArgs),
                color='k', alpha=0.1)
    ax.plot(modelX, obsY, marker='s', lw=0)
    if saveDIR is not None:
        fig1.savefig(path.join(saveDIR, 'cornerPlot.png'))
        fig2.savefig(path.join(saveDIR, 'modelPlot.png'))
    return fig1, fig2
