# -*- coding: utf-8 -*-
# /usr/bin/env python
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

"""calculate allowed telescope roll based on the allowed angle range
of starting from Y axis

"""


def telescopeRoll(PA, lowAngle, highAngle):
    """ calculate the telescope roll

    Parameters:
    PA: target position angle
    lowAngle: low end of the allowed range, measured from Y axis in anti-clock
        direction
    highAngle: high end of the allowed range, same way measured as lowAngle
    """
    lowAngle = lowAngle % 360
    highAngle = highAngle % 360
    roll1 = (135 + PA - highAngle) % 360
    roll2 = (135 + PA - lowAngle) % 360
    print('lower limit for the telescope rotation: {0:0.2f}'.format(roll1))
    print('higher limit for the telescope rotation: {0:0.2f}'.format(roll2))
