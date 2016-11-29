#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""use ffmpeg to convert mkv file to mp4 file
"""
from __future__ import division, absolute_import
from __future__ import print_function
import subprocess as sp
import sys
from os import path


def mkv2mp4(mkvFN, mp4FN):
    cnvtCMD = ['ffmpeg', '-i', mkvFN, '-c', 'copy', mp4FN]
    sp.call(cnvtCMD)


if __name__ == '__main__':
    mkvFN = path.expanduser(sys.argv[1])
    try:
        mp4FN = sys.argv[2]
    except IndexError:
        mp4FN = mkvFN.replace('.mkv', '.mp4')
    mkv2mp4(mkvFN, mp4FN)
