#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:31

@author: mvb
"""
import numpy as np

from model.pymodelDx import mpc_dynamic_vehicle_model
from model.kinematic_mpc_model import mpc_kinematic_vehicle_model
from simulator.integrate.systeminputhelper import getInput


def odeint_dx_dt(X,t):
    V = X[4:7]
    U = getInput(X[0])

    V_dt = mpc_dynamic_vehicle_model(V, U)

    c, s = np.cos(float(X[3])), np.sin(float(X[3]))
    R = np.array(((c, -s), (s, c)))
    Vabs = np.matmul(V[:2], R.transpose())

    return [1,Vabs[0],Vabs[1],V[2],V_dt[0],V_dt[1],V_dt[2]]


def solveivp_dynamic_dx_dt(t, X):
    V = [X[4][0], X[5][0], X[6][0]]
    U = getInput(X[0][0])

    print('t',t)
    V_dt = mpc_dynamic_vehicle_model(V, U)

    c, s = np.cos(float(X[3])), np.sin(float(X[3]))
    R = np.array(((c, -s), (s, c)))
    Vabs = np.matmul(V[:2], R.transpose())

    return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

def solveivp_kinematic_dx_dt(t, X):

    U = getInput(X[0][0])

    X_dt = mpc_kinematic_vehicle_model(X,U)

    return [1, X_dt[0], X_dt[1], X_dt[2], X_dt[3], X_dt[4], X_dt[5]]