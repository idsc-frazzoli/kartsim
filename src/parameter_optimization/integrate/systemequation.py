#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:31

@author: mvb
"""
import numpy as np

from parameter_optimization.model.marcsmodel_paramopt_own import marc_vehiclemodel
from parameter_optimization.integrate.systeminputhelper import getInput


def odeint_dx_dt(X,t):
    V = X[4:7]
    U = getInput(X[0])

    V_dt = marc_vehiclemodel(V, U)

    c, s = np.cos(float(X[3])), np.sin(float(X[3]))
    R = np.array(((c, -s), (s, c)))
    Vabs = np.matmul(V[:2], R.transpose())

    return [1,Vabs[0],Vabs[1],V[2],V_dt[0],V_dt[1],V_dt[2]]


def solveivp_dx_dt(t, X):
    V = np.array((X[4][0], X[5][0], X[6][0]))
    U = getInput(X[0][0])
    W = X[-3:]

    VU = np.append(V,U)

    V_dt = marc_vehiclemodel(VU,W)

    c, s = np.cos(float(X[3])), np.sin(float(X[3]))
    R = np.array(((c, -s), (s, c)))
    Vabs = np.matmul(V[:2], R.transpose())

    return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2], 0, 0, 0]