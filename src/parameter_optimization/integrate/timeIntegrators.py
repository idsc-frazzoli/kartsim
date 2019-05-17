#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:25 2019

@author: mvb
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp

from parameter_optimization.integrate.systemequation import odeint_dx_dt, solveivp_dx_dt
from parameter_optimization.integrate.systeminputhelper import setInput
from model.pymodelDx import marc_vehiclemodel

def odeIntegrator (X0, U, simStep, simIncrement):
    setInput(U)
    ts = np.linspace(0,simStep,int(simStep/simIncrement)+1)
    X1 = odeint(odeint_dx_dt, X0, ts)
    return X1


def odeIntegratorIVP(X0, U, simStep, simIncrement):
    setInput(U)
    t_eval = np.linspace(0, simStep, int(simStep / simIncrement) + 1)
    X1 = solve_ivp(solveivp_dx_dt, [0, simStep], X0, t_eval=t_eval, method='RK45', vectorized=True)

    return np.transpose(X1.y)


def euler (X0, simStep):
    x = float(X0[0])
    y = float(X0[1])
    theta = float(X0[2])
    vx = float(X0[3])
    vy = float(X0[4])
    vrot = float(X0[5])
    beta = float(X0[6])
    accRearAxle = float(X0[7])
    tv = float(X0[8])
#    [accX,accY,accRot] = eng.modelDx_pymod(vx,vy,vrot,beta,accRearAxle,tv, param, nargout=3) #This function only runs in Matlab Session. Shared Matlab session needed to access this function!
    [accX,accY,accRot] = marc_vehiclemodel(vx,vy,vrot,beta,accRearAxle,tv)
    vx = vx + accX * simStep
    vy = vy + accY * simStep
    vrot = vrot + accRot * simStep
    x = x + (vx * np.cos(theta) - vy * np.sin(theta)) * simStep
    y = y + (vy * np.cos(theta) + vx * np.sin(theta)) * simStep
    theta = theta + vrot * simStep
    X1 = np.array([[x,y,theta,vx,vy,vrot,beta,accRearAxle,tv]])
    return X1
    
    