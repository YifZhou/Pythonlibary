#! /usr/bin/env python
"""use shelve to save parameters
"""
from __future__ import division, absolute_import
from __future__ import print_function
import shelve

def save(outputFN=None, **saveParams):
    if outputFN is None:
        print('No save file name given')
        return 1
    db = shelve.open(outputFN)
    for key in saveParams:
        db[key] = saveParams[key]
    db.close()
    return 0
