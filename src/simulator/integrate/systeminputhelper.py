#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:40

@author: mvb
"""
from scipy.interpolate import interp1d
import numpy as np

def set_input_direct(U):
    global interpBETA, interpAB, interpTV
    interpBETA = interp1d(U[0], U[1], fill_value='extrapolate')
    interpAB = interp1d(U[0], U[2], fill_value='extrapolate')
    # interpTV = interp1d(U[0], U[3], fill_value='extrapolate')
    # interpBETA = [U[0], U[1]]
    # interpAB = [U[0], U[2]]
    interpTV = [U[0], U[3]]

def get_input_direct(time1):
    index = np.argmax(interpTV[0]>time1-0.1)
    # return [interpBETA(time1), interpAB(time1), interpTV(time1)]
    # return [interpBETA[1][index], interpAB[1][index], interpTV[1][index]]
    return [interpBETA(time1), interpAB(time1), interpTV[1][index]]

def set_input(U):
    global interpSTEER, interpBRAKE, interpMOTL, interpMOTR
    interpSTEER = interp1d(U[0], U[1], fill_value='extrapolate')
    interpBRAKE = interp1d(U[0], U[2], fill_value='extrapolate')
    interpMOTL = interp1d(U[0], U[3], fill_value='extrapolate')
    interpMOTR = interp1d(U[0], U[4], fill_value='extrapolate')

def get_input(time1):
    if abs(interpSTEER(time1)) < 0.01:
        return [0.01, interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]
    return [interpSTEER(time1), interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]

def setInputAccel(A):
    global interpX, interpY, interpTHETA
    interpX = interp1d(A[0], A[1], fill_value='extrapolate')
    interpY = interp1d(A[0], A[2], fill_value='extrapolate')
    interpTHETA = interp1d(A[0], A[3], fill_value='extrapolate')

def getInputAccel(time1):
    return [interpX(time1), interpY(time1), interpTHETA(time1)]
