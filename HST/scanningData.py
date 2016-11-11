#! /usr/bin/env python
from __future__ import print_function, division
import os
from os import path
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cPickle as pickle
import shelve
"""create data structure for scanning data
"""


def dqFilter(dq):
    flagList = [4, 8, 16, 32, 256, 512]
    DF = np.ones(dq.shape)
    for flag in flagList:
        DF[(dq // flag) % 2 == 1] = np.nan
    return DF


class scanFile(object):
    """scanning data structure

    """

    def __init__(self,
                 fileName,
                 fileDIR,
                 saveDIR,
                 dqMask,
                 skyMask,
                 removeSky=True,
                 arraySize=256):
        super(scanFile, self).__init__()
        self.dqMask = dqMask
        self.skyMask = skyMask
        self.fileDIR = fileDIR
        self.saveDIR = saveDIR
        self.removeSky = removeSky
        self.arraySize = arraySize
        self.imaFN = path.join(fileDIR, fileName)
        imaHeader = fits.getheader(self.imaFN, 0)
        self.rootName = imaHeader['ROOTNAME']
        self.nSamp = imaHeader['NSAMP']
        self.expTime = imaHeader['EXPTIME']
        ima1Header = fits.getheader(self.imaFN, 'sci', 1)
        self.unit = ima1Header['BUNIT']
        # unit: counts, specifically for scanning data file
        self.imaDataCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.imaSampTime = np.zeros(self.nSamp)
        # unit: counts
        self.imaDataDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.readIma()

    def readIma(self):
        with fits.open(self.imaFN) as f:
            for i in xrange(self.nSamp):
                # ima file is stored backwardly
                # in my data, they are saved in a normal direction
                # so reverse the order for ima files
                self.imaDataCube[:, :, i] = f['sci', self.nSamp-i].\
                                            data[5:5+self.arraySize,
                                                 5:5+self.arraySize] * self.dqMask
                self.imaSampTime[i] = f['sci', self.nSamp - i].header[
                    'SAMPTIME']
                if self.unit == 'ELECTRONS/S':
                    self.imaDataCube[:, :, i] = self.imaDataCube[:, :, i] * self.imaSampTime[i]
                if i > 0:
                    self.imaDataCube[:, :, i] = self.imaDataCube[:, :, i] + self.imaDataCube[:, :, 0]
            for i in xrange(self.nSamp - 1):
                self.imaDataDiff[:, :, i] =\
                    self.imaDataCube[:, :, i+1] - self.imaDataCube[:, :, i]
            if self.removeSky:
                self.removeSky()

    def removeSky(self, klipthresh=2):
        self.skyValue = np.ones(self.nSamp - 1)
        for i in xrange(self.nSamp - 1):
            skyMask_i = self.skyMask.copy()
            for j in xrange(10):
                sigma = np.nanstd(skyMask_i * self.imaDataDiff[:, :, i])
                med = np.nanmedian(skyMask_i * self.imaDataDiff[:, :, i])
                sigmaKlipID = np.where((
                    self.imaDataDiff[:, :, i] > klipthresh * sigma + med) | (
                        self.imaDataDiff[:, :, i] < med - klipthresh * sigma))
                if len(sigmaKlipID) == 0:
                    break
                skyMask_i[sigmaKlipID[0], sigmaKlipID[1]] = np.nan
            self.skyValue[i] = np.nanmedian(skyMask_i *
                                            self.imaDataDiff[:, :, i])
            self.imaDataDiff[:, :, i] = self.imaDataDiff[:, :, i] -\
                self.skyValue[i]

    def pixelLightCurve(self, x, y):
        return self.imaDataCube[y, x, :]

    def pixelDiffLightCurve(self, x, y):
        return self.imaDataDiff[y, x, :]

    def pixelCount(self, x, y, nSampStart=1):
        """total count for specific pixels, the count from 0 to 1st read by default is discarded"""
        if np.isnan(self.dqMask[y, x]):
            return np.nan
        else:
            return np.nansum(self.imaDataDiff[y, x, nSampStart:])

    def plotSampleImage(self, nSamp):
        if nSamp >= self.nSamp:
            print('Maximum Sample number is {0}'.format(self.nSamp - 1))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(self.imaDataCube[:, :, nSamp],
                             vmin=-200,
                             vmax=200,
                             origin='lower')
            fig.colorbar(cax)
            ax.text(0.02,
                    0.02,
                    'Sample: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp),
                    transform=ax.transAxes,
                    backgroundcolor='0.9')
            ax.text(0.02,
                    0.08,
                    'Sample Time: {0:.2f}'.format(self.imaSampTime[nSamp]),
                    transform=ax.transAxes,
                    backgroundcolor='0.9')
        return fig

    def plotDiffImage(self, nSamp):
        if nSamp >= self.nSamp:
            print('Maximum Diff Sample number is {0}'.format(self.nSamp - 2))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(self.imaDataDiff[:, :, nSamp],
                             vmin=-200,
                             vmax=200,
                             origin='lower')
            fig.colorbar(cax)
            ax.text(0.02,
                    0.02,
                    'Diff No.: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp - 1),
                    transform=ax.transAxes,
                    backgroundcolor='0.9')
            ax.text(0.02,
                    0.08,
                    'Diff Samp Time: {0:.2f}-{1:.2f}'.format(
                        self.imaSampTime[nSamp + 1], self.imaSampTime[nSamp]),
                    transform=ax.transAxes,
                    backgroundcolor='0.9')
        return fig

    def save(self, rootName=None):
        if rootName is None:
            rootName = self.rootName
        if not os.path.exists(self.saveDIR):
            os.makedirs(self.saveDIR)
        saveFN = path.join(self.saveDIR, rootName + '.pickle')
        with open(saveFN, 'wb') as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)


class scanData(object):
    """the whole dataset for the scanning file

    """

    def __init__(self,
                 infoFile,
                 fileDIR,
                 saveDIR,
                 dqFN,
                 skyFN,
                 restore=False,
                 restoreDIR=None):
        super(scanData, self).__init__()
        self.info = pd.read_csv(infoFile,
                                parse_dates=True,
                                index_col='Datetime')
        self.info.sort_values('Time')
        self.info = self.info[(self.info['Filter'] == 'G141')]
        self.dqMask = dqFilter(fits.getdata(dqFN, 0))
        self.skyMask = fits.getdata(skyFN, 0)
        self.scanFileList = []
        self.saveDIR = saveDIR
        self.time = self.info['Time'].values
        self.orbit = self.info['Orbit'].values
        self.expTime = self.info['Exp Time'].values[0]
        if restore:
            if restoreDIR is None:
                restoreDIR = saveDIR
            for fn in self.info['File Name']:
                with open(path.join(restoreDIR, fn.replace('_ima.fits',
                                                           '.pickle'))) as pkf:
                    self.scanFileList.append(pickle.load(pkf))
        else:
            for fn in self.info['File Name']:
                self.scanFileList.append(scanFile(fn, fileDIR, saveDIR,
                                                  self.dqMask, self.skyMask))

    def showExampleImage(self, n=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.scanFileList[n].imaDataCube[:, :, -1] -
                         np.nanmin(self.scanFileList[n].imaDataCube[:, :, -1]),
                         origin='lower',
                         norm=LogNorm(),
                         cmap='viridis')
        fig.colorbar(cax)
        return fig

    def pixelLightCurve(self, x, y, plot=False):
        if np.isnan(self.dqMask[y, x]):
            print('Speficied Pixel is a bad pixel')
            return None
        else:
            lc = np.array([sf.pixelCount(x, y) for sf in self.scanFileList])
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(self.time, lc, 'o')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Counts')
                ax.text(0.05,
                        0.05,
                        'x={0}, y={1}'.format(x, y),
                        transform=ax.transAxes)
            return lc

    def columnLightCurve(self, x, yRange, plot=False):
        """draw the light curve from a column"""
        lc = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            lc[i] = np.nanmean(
                [sf.pixelCount(x, y) for y in xrange(yRange[0], yRange[1])])
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.time, lc, 'o')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.text(0.05,
                    0.05,
                    'x={0}, y=({1}, {2})'.format(x, yRange[0], yRange[1]),
                    transform=ax.transAxes)
        return lc

    def plotSkyTrend(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, sf in enumerate(self.scanFileList):
            ax.plot(self.time[i] + sf.imaSampTime[2:], sf.skyValue[1:], 'bo')
            ax.plot(self.time[i] + sf.imaSampTime[2:],
                    sf.skyValue[1:],
                    '-',
                    color='0.8')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Sky Count [$\mathsf{e^-}$]')

    def skyTrend(self):
        t = []
        sky = []
        for i, sf in enumerate(self.scanFileList):
            t.append(self.time[i] + sf.imaSampTime[2:])
            sky.append(sf.skyValue[1:])
        return np.concatenate(t), np.concatenate(sky)

    def save(self):
        for sf in self.scanFileList:
            sf.saveDIR = self.saveDIR
            sf.save()


if __name__ == '__main__':
    # visits = range(1, 16)
    visits = range(1, 16)
    xStarts = np.array([65, 70, 65, 65, 65, 70, 70, 70, 70, 70, 70, 70, 70, 70,
                        75])
    xEnds = np.array([185, 185, 180, 180, 180, 180, 180, 185, 185, 190, 190,
                      195, 185, 185, 185])
    yStarts = np.array([155, 150, 155, 155, 155, 150, 150, 155, 145, 150, 145,
                        150, 150, 150, 145])
    yEnds = np.array([230, 235, 230, 220, 230, 240, 235, 230, 235, 235, 235,
                      235, 235, 235, 235])
    fileDIR = path.expanduser('~/Documents/GJ1214/DATA')

    for i, visit in enumerate(visits):
        infoFN = path.expanduser('~/Documents/GJ1214/'
                                 'GJ1214_visit_{0:02d}_fileInfo.csv'.format(
                                     visit))
        saveDIR = path.expanduser(
            '~/Documents/GJ1214/scanningData//pickle{0:02d}_SkyMask'.format(visit))
        dqFN = path.expanduser('~/Documents/GJ1214/scanningData/commonDQ.fits')
        skyFN = path.expanduser(
            '~/Documents/GJ1214/scanningData/skyMask_visit_{0:02d}.fits'.format(visit))
        sd = scanData(infoFN,
                      fileDIR,
                      saveDIR,
                      dqFN,
                      skyFN,
                      restore=True,
                      restoreDIR=saveDIR)
        # collect light curves
        xList = range(xStarts[i], xEnds[i])
        LCmatrix = np.zeros((len(xList), len(sd.time)))
        for j, x in enumerate(xList):
            LCmatrix[j, :] = sd.columnLightCurve(x, [yStarts[i], yEnds[i]])
        db = shelve.open(path.join(
            saveDIR, 'LCmatrix_visit_{0:02d}.shelve'.format(visit)))
        db['LCmatrix'] = LCmatrix
        db['time'] = sd.time
        db['xList'] = xList
        db['orbit'] = sd.orbit
        db['expTime'] = sd.expTime
        db.close()
        db = shelve.open(path.join(
            saveDIR, 'LCmatrix_visit_{0:02d}_sky.shelve'.format(visit)))
        t, sky = sd.skyTrend()
        db['time'] = t
        db['sky'] = sky
        db.close()
