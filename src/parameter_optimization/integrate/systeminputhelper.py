#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:40

@author: mvb
"""
from scipy.interpolate import interp1d
import numpy as np

def setInput(U):
    global interpBETA, interpAB, interpTV
    interpBETA = interp1d(U[0], U[1], fill_value='extrapolate')
    interpAB = interp1d(U[0], U[2], fill_value='extrapolate')
    interpTV = interp1d(U[0], U[3], fill_value='extrapolate')

def getInput(time1):
    return np.array((interpBETA(time1), interpAB(time1), interpTV(time1)))