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
    index = np.argmax(interpTV[0] > time1 - 0.1)
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
    steering_angle1 = interpSTEER(time1)
    if abs(steering_angle1) < 0.005:
        if steering_angle1 < 0:
            return [-0.005, interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]
        else:
            return [0.005, interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]
    return [steering_angle1, interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]


def setInputAccel(A):
    global interpX, interpY, interpTHETA
    interpX = interp1d(A[0], A[1], fill_value='extrapolate')
    interpY = interp1d(A[0], A[2], fill_value='extrapolate')
    interpTHETA = interp1d(A[0], A[3], fill_value='extrapolate')


def getInputAccel(time1):
    return [interpX(time1), interpY(time1), interpTHETA(time1)]


def set_kinematic_input(U):
    global interpSTEER, interpBRAKE, interpMOTL, interpMOTR, interpDBETA
    interpSTEER = interp1d(U[0], U[1], fill_value='extrapolate')
    interpBRAKE = interp1d(U[0], U[2], fill_value='extrapolate')
    interpMOTL = interp1d(U[0], U[3], fill_value='extrapolate')
    interpMOTR = interp1d(U[0], U[4], fill_value='extrapolate')
    # turning_angle = -0.63 * np.power(U[1], 3) + 0.94 * U[1]
    # dturning_angle = (turning_angle[1:] - turning_angle[:-1]) / (U[0][1:] - U[0][:-1])
    # interpDBETA = interp1d(np.array(U[0][:-1]) + (U[0][1] - U[0][0]) / 2.0, dturning_angle, fill_value='extrapolate')


def set_kinematic_input_direct(U):
    global interpBETA, interpAB, interpTV, interpDBETA
    interpBETA = interp1d(U[0], U[1], fill_value='extrapolate')
    interpAB = interp1d(U[0], U[2], fill_value='extrapolate')
    interpTV = [U[0], U[3]]
    dturning_angle = (U[1][1:] - U[1][:-1]) / (U[0][1:] - U[0][:-1])
    if len(dturning_angle) > 1:
        interpDBETA = interp1d(U[0][:-1], dturning_angle, fill_value='extrapolate')
    else:
        interpDBETA = interp1d(U[0], [dturning_angle[0], dturning_angle[0]], fill_value='extrapolate')


def get_kinematic_input(time1):
    return [interpSTEER(time1), interpBRAKE(time1), interpMOTL(time1), interpMOTR(time1)]


def get_kinematic_input_direct(time1):
    index = np.argmax(interpTV[0] > time1 - 0.1)
    # return [interpBETA(time1), interpAB(time1), interpTV(time1)]
    # return [interpBETA[1][index], interpAB[1][index], interpTV[1][index]]
    return [float(interpBETA(time1)), float(interpAB(time1)), float(interpTV[1][index])]
