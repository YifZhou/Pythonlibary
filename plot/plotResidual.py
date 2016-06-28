#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def plotResidual(xData, yData, models,
                 xlabel='',
                 ylabel='',
                 yThresh=0.01,
                 yNorm=None,  # normalization factor for y
                 modelLabels=None,
                 axes=None,
                 marker='o',
                 lw=2,
                 plotkw={}):
    if yNorm is None:
        yNorm = np.nanmedian(yData)
    if axes is None:
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.3, 0.8, 0.6))
        ax_res = fig.add_axes((0.1, 0.1, 0.8, 0.2), sharex=ax)
    ax.plot(xData, 100 * (yData / yNorm - 1),
            marker=marker, mfc='k', label='data', linestyle='', zorder=3,
            **plotkw)
    if modelLabels is None:
        modelLabels = ['Model {0:d}'.format(i) for i in xrange(len(models))]
    for label, model in zip(modelLabels, models):
        l = ax.plot(xData, 100 * (model / yNorm - 1), lw=lw, label=label,
                    **plotkw)
        ax_res.plot(xData, 100 * (yData - model) / yNorm,
                    marker=marker,
                    color=l[0].get_color(), linestyle='', **plotkw)
    # different scale
    # ax.set_ylim(ylim)
    axy2 = ax.twinx()
    axy2.set_ylim((np.array(ax.get_ylim()) / 100 + 1) * yNorm)
    axy2.set_yticks((ax.get_yticks() / 100 + 1) * yNorm)
    ax_res_y2 = ax_res.twinx()
    ax_res.set_ylim([-yThresh*100, yThresh*100])
    ax_res_y2.set_ylim(np.array(ax_res.get_ylim()) / 100 * yNorm)
    ax_res_y2.set_yticks((ax_res.get_yticks() / 100) * yNorm)
    for xlabel in ax.get_xticklabels():
        xlabel.set_visible(False)
    for xlabel in axy2.get_xticklabels():
        xlabel.set_visible(False)
    ax.get_yticklabels()[0].set_visible(False)
    axy2.get_yticklabels()[0].set_visible(False)
    ax_res.get_yticklabels()[-1].set_visible(False)
    ax_res_y2.get_yticklabels()[-1].set_visible(False)
    ax.legend(loc='lower right')
    ax_res.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, [ax, axy2, ax_res, ax_res_y2]

if __name__ == '__main__':
    from functions import gaussian
    from lmfit.models import GaussianModel
    x = np.linspace(-3, 3, 200)
    y = gaussian(x, sigma=0.5) * (1 + np.random.normal(0, 0.1, len(x)))
    gmod = GaussianModel()
    para = gmod.guess(y, x=x)
    gfit = gmod.fit(y, para, x=x)
    models = [gfit.init_fit, gfit.best_fit]
    plt.close('all')
    fig, axes = plotResidual(x, y, models, yNorm=1.0, yThresh=0.1)
    axes[0].set_ylabel('relative [%]')
    axes[1].set_ylabel('absolute')
    axes[2].set_xlabel('xxx')
    plt.show()
