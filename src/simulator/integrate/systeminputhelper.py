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
    # interpTV = interp1d(U[0], U[3], fill_value='extrapolate')
    # interpBETA = [U[0], U[1]]
    # interpAB = [U[0], U[2]]
    interpTV = [U[0], U[3]]

def getInput(time1):
    index = np.argmax(interpTV[0]>time1-0.1)
    # return [interpBETA(time1), interpAB(time1), interpTV(time1)]
    # return [interpBETA[1][index], interpAB[1][index], interpTV[1][index]]
    # print([interpBETA(time1), interpAB(time1), interpTV[1][index]])
    return [interpBETA(time1), interpAB(time1), interpTV[1][index]]