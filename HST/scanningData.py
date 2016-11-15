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
from scipy.signal import medfilt
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from fit import linearFit
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
                 flat,
                 ROI=[0, 256, 0, 256],
                 arraySize=256,
                 scanDirect=0):
        """ROI defines the region of interest for cosmic Ray removal
        """
        super(scanFile, self).__init__()
        self.dqMask = dqMask.copy()
        self.skyMask = skyMask.copy()
        self.flat = flat
        self.fileDIR = fileDIR
        self.saveDIR = saveDIR
        self.ROI = ROI
        self.arraySize = arraySize
        self.scanDirect = scanDirect
        self.imaFN = path.join(fileDIR, fileName)
        imaHeader = fits.getheader(self.imaFN, 0)
        self.rootName = imaHeader['ROOTNAME']
        self.nSamp = imaHeader['NSAMP']
        self.expTime = imaHeader['EXPTIME']
        # unit: counts, specifically for scanning data file
        self.imaDataCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.errCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.imaSampTime = np.zeros(self.nSamp)
        # unit: counts
        self.imaDataDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.errDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.skyVar = np.zeros(self.nSamp - 1)  # error of sky subtraction
        self.scanLength = None
        self.scanStart = None
        self.readIma()
        # self.calScanLength()

    def readIma(self):
        with fits.open(self.imaFN) as f:
            self.removeCR(f)
            for i in xrange(self.nSamp):
                # ima file is stored backwardly
                # in my data, they are saved in a normal direction
                # so reverse the order for ima files
                self.imaDataCube[:, :, i] = f['sci', self.nSamp-i].\
                                            data[5:5+self.arraySize,
                                                 5:5+self.arraySize] / self.flat * self.dqMask
                self.errCube[:, :, i] = f['err', self.nSamp-i].\
                                            data[5:5+self.arraySize,
                                                 5:5+self.arraySize] * self.dqMask
                self.imaSampTime[i] = f['sci', self.nSamp - i].header[
                    'SAMPTIME']
            for i in xrange(1, self.nSamp):
                self.imaDataCube[:, :, i] = self.imaDataCube[:, :, i] + \
                                            self.imaDataCube[:, :, 0]
            for i in xrange(self.nSamp - 1):
                self.imaDataDiff[:, :, i] =\
                    self.imaDataCube[:, :, i+1] - self.imaDataCube[:, :, i]
                self.errDiff[:, :, i] =\
                    np.sqrt(self.errCube[:, :, i+1]**2 + self.errCube[:, :, i]**2)
            self.removeSky()

    def removeCR(self, fitsFile, sigmaLevel=7):
        # use a median filter to smooth the region of interest first
        # if self.rootName == 'ibxy02jpq':
        #     import ipdb;ipdb.set_trace()
        im = fitsFile['sci', 1].data[5:5 + self.arraySize, 5:5 +
                                     self.arraySize] / self.flat * self.dqMask
        imROI = im[self.ROI[2]:self.ROI[3], self.ROI[0]:self.ROI[1]]
        # use a 5 pixel sized median filter to remove hot pixels
        err = fitsFile['err', 1].data[5:5 + self.arraySize, 5:5 +
                                      self.arraySize] / self.flat * self.dqMask
        errROI = err[self.ROI[2]:self.ROI[3], self.ROI[0]:self.ROI[1]]
        diff1 = np.abs(imROI - medfilt(imROI, [1, 7]))/errROI
        diff2 = np.abs(imROI - medfilt(imROI, [7, 1]))/errROI
        diff = np.minimum(diff1, diff2)
        yCR, xCR = np.where(diff > sigmaLevel)
        self.dqMask[self.ROI[2]+yCR, self.ROI[0]+xCR] = np.nan
        dqROI = self.dqMask[self.ROI[2]:self.ROI[3], self.ROI[0]:self.ROI[1]]
        self.scanRate = np.nanmean((imROI*dqROI)[:, 20:-20], axis=1) /\
                          np.median(np.nanmean((imROI*dqROI)[:, 20:-20], axis=1))
        self.scanDQIndex = np.where(np.abs(self.scanRate - 1) > 0.04)[0]  # find scan rate anomaly
        # for scanDQi in self.scanDQIndex:
        #     self.dqMask[self.ROI[2]+scanDQi-5:self.ROI[2]+scanDQi+6,
        #                 self.ROI[0]:self.ROI[1]] = np.nan
        print("file:{0}  {1}/{2} cosmic ray found".format(self.rootName, len(yCR), imROI.size))
        print("file:{0} {1} lines large scanrate".format(self.rootName, len(self.scanDQIndex)))

    def removeSky(self, klipthresh=5):
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
            self.skyVar[i] = np.nanstd(skyMask_i*self.imaDataDiff[:, :, i])**2
            self.imaDataDiff[:, :, i] = self.imaDataDiff[:, :, i] -\
                self.skyValue[i]

    def pixelLightCurve(self, x, y):
        return self.imaDataCube[y, x, :]

    def pixelDiffLightCurve(self, x, y):
        return self.imaDataDiff[y, x, :]

    def pixelCount(self, x, y, nSampStart=1):
        """total count for specific pixels,
        the count from 0 to 1st read by default is discarded"""
        if np.isnan(self.dqMask[y, x]):
            return np.nan
        else:
            return np.nansum(self.imaDataDiff[y, x, nSampStart:])

    def pixelError(self, x, y):
        if np.isnan(self.dqMask[y, x]):
            return np.nan
        else:
            return np.sqrt(self.errCube[y, x, -1]**2 + self.skyVar.sum())

    def columnCount(self, x, yRange, nSampStart=1):
        column = np.array([self.pixelCount(x, y, nSampStart) for y in xrange(yRange[0], yRange[1])])
        return np.nanmean(column)
        # return np.nanmedian(column)

    def columnError(self, x, yRange, nSampStart=1):
        error = np.array([self.pixelError(x, y) for y in xrange(yRange[0], yRange[1])])
        return np.sqrt(np.nanmean(error**2) / (len(error) - np.isnan(error).sum()))

    def white(self, yRange=None):
        """return white count and errors"""
        if yRange is None:
            yRange = [self.ROI[2], self.ROI[3]]
        nColumn = self.ROI[1] - self.ROI[0]
        counts = np.zeros(nColumn)
        errors = np.zeros(nColumn)
        for i, x in enumerate(xrange(self.ROI[0], self.ROI[1])):
            counts[i] = self.columnCount(x, yRange)
            errors[i] = self.columnError(x, yRange)
        count = np.sum(counts / errors**2) / np.sum(1 / errors**2)
        error = 1 / np.sum(1 / errors**2)
        return count, error

    def plotSampleImage(self, nSamp):
        if nSamp >= self.nSamp:
            print('Maximum Sample number is {0}'.format(self.nSamp - 1))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(
                self.imaDataCube[:, :, nSamp],
                vmin=-200,
                vmax=200,
                origin='lower')
            fig.colorbar(cax)
            ax.text(
                0.02,
                0.02,
                'Sample: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp),
                transform=ax.transAxes,
                backgroundcolor='0.9')
            ax.text(
                0.02,
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
            cax = ax.matshow(
                self.imaDataDiff[:, :, nSamp],
                vmin=-200,
                vmax=200,
                origin='lower')
            fig.colorbar(cax)
            ax.text(
                0.02,
                0.02,
                'Diff No.: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp - 1),
                transform=ax.transAxes,
                backgroundcolor='0.9')
            ax.text(
                0.02,
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

    def calScanLength(self, level=5):
        """calculate the scanning length and scanning starting point
        the scanning starting point and scanning length are calculated using cubic interpolation at multiple level
        """
        self.scanLengthLevel = level
        self.scanLength = np.zeros(level)
        self.scanStart = np.zeros(level)
        levels = np.linspace(0.1, 0.9, level)
        x = np.arange(self.arraySize)
        for i, l in enumerate(levels):
            f_interp = interp(x, self.scanRate - l)
            roots = f_interp.roots()
            self.scanLength[i] = abs(roots[1] - roots[0])
            if self.scanDirect == 0:
                self.scanStart[i] = min(roots)
            else:
                self.scanStart[i] = max(roots)


class scanData(object):
    """the whole dataset for the scanning file

    """

    def __init__(self,
                 infoFile,
                 fileDIR,
                 saveDIR,
                 dqFN,
                 skyFN,
                 flatFN,
                 ROI,
                 arraySize=256,
                 restore=False,
                 twoDirect=False,
                 restoreDIR=None):
        super(scanData, self).__init__()
        self.info = pd.read_csv(
            infoFile, parse_dates=True, index_col='Datetime')
        self.info.sort_values('Time')
        self.info = self.info[(self.info['Filter'] == 'G141')]
        self.dqMask = dqFilter(fits.getdata(dqFN, 0))
        self.skyMask = fits.getdata(skyFN, 0)
        self.flat = fits.getdata(flatFN, 0)
        # self.skyMask = np.full(self.dqMask.shape, np.nan)
        # self.skyMask[ROI[2]:ROI[3], 10:35] = 1
        # self.skyMask = self.skyMask * self.dqMask
        self.arraySize = arraySize
        self.ROI = ROI
        self.scanFileList = []
        self.saveDIR = saveDIR
        self.time = self.info['Time'].values
        self.orbit = self.info['Orbit'].values
        self.expTime = self.info['Exp Time'].values[0]
        if twoDirect:
            self.scanDirect = self.info['scanDirect'].values
        self.twoDirectScale = 1.0

        if restore:
            if restoreDIR is None:
                restoreDIR = saveDIR
            for fn in self.info['File Name']:
                with open(
                        path.join(restoreDIR, fn.replace('_ima.fits',
                                                         '.pickle'))) as pkf:
                    self.scanFileList.append(pickle.load(pkf))
        else:
            for i, fn in enumerate(self.info['File Name']):
                self.scanFileList.append(
                    scanFile(fn, fileDIR, saveDIR, self.dqMask, self.skyMask, self.flat,
                             self.ROI, arraySize=self.arraySize,
                             scanDirect=self.scanDirect[i]))

    def showExampleImage(self, n=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(
            self.scanFileList[n].imaDataCube[:, :, -1] -
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
            lc_err = np.array([sf.pixelError(x, y) for sf in self.scanFileList])
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(self.time, lc, yerr=lc_err, fmt='o', ls='')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Counts')
                ax.text(
                    0.05,
                    0.05,
                    'x={0}, y={1}'.format(x, y),
                    transform=ax.transAxes)
            return lc, lc_err

    def columnLightCurve(self, x, yRange, plot=False):
        """draw the light curve from a column"""
        lc = np.zeros(len(self.scanFileList))
        lc_err = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            lc[i] = sf.columnCount(x, yRange)
            lc_err[i] = sf.columnError(x, yRange)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time, lc, yerr=lc_err, fmt='o', ls='')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.text(
                0.05,
                0.05,
                'x={0}, y=({1}, {2})'.format(x, yRange[0], yRange[1]),
                transform=ax.transAxes)
        return lc, lc_err

    def whiteLC(self, yRange=None, plot=False):
        wlc = np.zeros(len(self.scanFileList))
        wlc_err = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            count, error = sf.white(yRange=yRange)
            wlc[i] = count
            wlc_err[i] = error
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time[self.scanDirect == 0], wlc[self.scanDirect == 0],
                        yerr=wlc_err[self.scanDirect == 0], fmt='.', ls='', label='upwards')
            ax.errorbar(self.time[self.scanDirect == 1], wlc[self.scanDirect == 1],
                        yerr=wlc_err[self.scanDirect == 1], fmt='.', ls='', label='downwards')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.set_title('White Light Curve')
            ax.legend()
        return wlc, wlc_err

    def plotSkyTrend(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, sf in enumerate(self.scanFileList):
            ax.plot(self.time[i] + sf.imaSampTime[2:], sf.skyValue[1:], 'bo')
            ax.plot(
                self.time[i] + sf.imaSampTime[2:],
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

    def plotScanRate(self):
        fig, ax = plt.subplots()
        yPixels = np.arange(self.ROI[2], self.ROI[3])
        for i, sf in enumerate(self.scanFileList):
            if self.scanDirect[i] == 0:
                c = 'r'
            else:
                c = 'b'
            ax.plot(yPixels, sf.scanRate, lw=0.5, ls='steps',
                    color=c, alpha=0.8)
        ax.set_xlabel('Y')
        ax.set_ylabel('scan rate')
        return fig

    def save(self):
        for sf in self.scanFileList:
            sf.saveDIR = self.saveDIR
            sf.save()

    def calcTwoDirectScale(self, yRange=None, nStart=3):
        wlc, wlc_err = self.whiteLC(yRange=yRange)
        scanCorrFactList = np.zeros(2)
        for i, orbit in enumerate((0, 1)):
            # remove first several exposures affected by ramp effect
            lc_upp = wlc[(self.orbit == orbit) & (self.scanDirect == 0)]
            lc_down = wlc[(self.orbit == orbit) & (self.scanDirect == 1)]
            scanCorrFactList[i] = (lc_upp[nStart:]).mean() / (lc_down[nStart:]).mean()
        return scanCorrFactList.mean()

    def plotScanLength(self, iLevel=3):
        scanLength = []
        scanStart = []
        for sf in self.scanFileList:
            scanLength.append(sf.scanLength[iLevel])
            if sf.scanDirect == 0:
                scanStart.append(sf.scanStart[iLevel])
            else:
                scanStart.append(511 - sf.scanStart[iLevel])
        fig, ax = plt.subplots()
        ax.plot(scanStart, scanLength, '.')
        ax.set_xlabel('Scanning Start [px]')
        ax.set_ylabel('Scanning Length [px]')
        b, m, _, _, _ = linearFit(np.array(scanStart), np.array(scanLength))
        x = np.array([min(scanStart), max(scanStart)])
        ax.plot(x, b + m*x, '--')
        return fig


if __name__ == '__main__':
    pass
    # visits = range(1, 16)
    # visits = range(1, 16)
    # xStarts = np.array([65, 70, 65, 65, 65, 70, 70, 70, 70, 70, 70, 70, 70, 70,
    #                     75])
    # xEnds = np.array([185, 185, 180, 180, 180, 180, 180, 185, 185, 190, 190,
    #                   195, 185, 185, 185])
    # yStarts = np.array([155, 150, 155, 155, 155, 150, 150, 155, 145, 150, 145,
    #                     150, 150, 150, 145])
    # yEnds = np.array([230, 235, 230, 220, 230, 240, 235, 230, 235, 235, 235,
    #                   235, 235, 235, 235])
    # fileDIR = path.expanduser('~/Documents/GJ1214/DATA')

    # for i, visit in enumerate(visits):
    #     infoFN = path.expanduser('~/Documents/GJ1214/'
    #                              'GJ1214_visit_{0:02d}_fileInfo.csv'.format(
    #                                  visit))
    #     saveDIR = path.expanduser(
    #         '~/Documents/GJ1214/scanningData//pickle{0:02d}_SkyMask'.format(visit))
    #     dqFN = path.expanduser('~/Documents/GJ1214/scanningData/commonDQ.fits')
    #     skyFN = path.expanduser(
    #         '~/Documents/GJ1214/scanningData/skyMask_visit_{0:02d}.fits'.format(visit))
    #     sd = scanData(infoFN,
    #                   fileDIR,
    #                   saveDIR,
    #                   dqFN,
    #                   skyFN,
    #                   restore=True,
    #                   restoreDIR=saveDIR)
    #     # collect light curves
    #     xList = range(xStarts[i], xEnds[i])
    #     LCmatrix = np.zeros((len(xList), len(sd.time)))
    #     for j, x in enumerate(xList):
    #         LCmatrix[j, :] = sd.columnLightCurve(x, [yStarts[i], yEnds[i]])
    #     db = shelve.open(path.join(
    #         saveDIR, 'LCmatrix_visit_{0:02d}.shelve'.format(visit)))
    #     db['LCmatrix'] = LCmatrix
    #     db['time'] = sd.time
    #     db['xList'] = xList
    #     db['orbit'] = sd.orbit
    #     db['expTime'] = sd.expTime
    #     db.close()
    #     db = shelve.open(path.join(
    #         saveDIR, 'LCmatrix_visit_{0:02d}_sky.shelve'.format(visit)))
    #     t, sky = sd.skyTrend()
    #     db['time'] = t
    #     db['sky'] = sky
    #     db.close()
