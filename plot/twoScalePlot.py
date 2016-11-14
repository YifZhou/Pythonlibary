#! /usr/bin/env python
"""make a plot use two scales
relative scale using y1 (left)
and absolute scale using y2 (right)
"""
from __future__ import division, absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


def twoScalePlot(x,
                 y,
                 scaleFactor=1,
                 xlabel='',
                 y1label='',
                 y2label='',
                 ax=None,
                 **plotkw):
    if ax is None:
        fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(x, y / scaleFactor, **plotkw)
    ax2.set_ylim(np.array(ax.get_ylim()) * scaleFactor)
    # ax2.set_yticks(ax.get_yticks())
    # ax2.set_yticklabels(
    #     ['{0:.2f}'.format(ytick * scaleFactor) for ytick in ax.get_yticks()])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y1label)
    ax2.set_ylabel(y2label)
    return ax.figure

if __name__ == '__main__':
    x = np.linspace(0, 3 * np.pi, 300)
    y = 200*np.sin(x + np.random.randn(1))
    fig, ax = plt.subplots()
    fig = twoScalePlot(x, y, scaleFactor=200, ax=ax,
                    xlabel='t', y1label='rel', y2label='abs', lw=2, marker='+')
