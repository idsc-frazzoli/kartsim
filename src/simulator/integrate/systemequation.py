#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:31

@author: mvb
"""
import numpy as np

from model.pymodelDx import marc_vehiclemodel
from simulator.integrate.systeminputhelper import getInput

def odeint_dx_dt(X,t):
    theta = float(X[3])
    vx = float(X[4])
    vy = float(X[5])
    vrot = float(X[6])

    beta, accRearAxle, tv = getInput(X[0])

    [accX,accY,accRot] = marc_vehiclemodel(vx, vy, vrot, beta, accRearAxle, tv)
    vxAbs = vx * np.cos(theta) - vy * np.sin(theta)
    vyAbs = vy * np.cos(theta) + vx * np.sin(theta)

    return [1,vxAbs,vyAbs,vrot,accX,accY,accRot,0,0,0]


def solveivp_dx_dt(t, X):
    # print('X', X)
    theta = float(X[3])
    vx = float(X[4])
    vy = float(X[5])
    vrot = float(X[6])

    beta, accRearAxle, tv = getInput(X[0])

    [accX, accY, accRot] = marc_vehiclemodel(vx, vy, vrot, beta[0], accRearAxle[0], tv[0])

    vxAbs = vx * np.cos(theta) - vy * np.sin(theta)
    vyAbs = vy * np.cos(theta) + vx * np.sin(theta)

    return [1, vxAbs, vyAbs, vrot, accX, accY, accRot]