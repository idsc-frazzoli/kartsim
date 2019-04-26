#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:25 2019

@author: mvb
"""
from scipy.integrate import odeint, solve_ivp
import numpy as np

from simulator.pymodelDx import pymodelDx
from scipy.interpolate import interp1d

import time

def setInput(U):
    global interpBETA, interpAB, interpTV
    interpBETA = interp1d(U[0], U[1], fill_value='extrapolate')
    interpAB = interp1d(U[0], U[2], fill_value='extrapolate')
    interpTV = interp1d(U[0], U[3], fill_value='extrapolate')

def getInput(time1):
    # print(U_array[0], U_array[1],time1)
    # print(interpBETA(time1))
    return interpBETA(time1), interpAB(time1), interpTV(time1)

def MATLABvehicleModel (vx,vy,vrot,beta,accRearAxle,tv):
    #debug params
    # B = 4.0
    # C = 1.7
    # D = 0.7*9.81
    # Cf = 0.15
    # B1 = B
    # B2 = B
    # C1 = C
    # C2 = C
    # D1 = 0.8*D
    # D2 = D
    # maxA = D*0.9

    #optimal parameters
    B1 = 15
    C1 = 1.1
    D1 = 9.4
    B2 = 5.2
    C2 = 1.4
    D2 = 10.4
    Cf = 0.3

    param = [B1,C1,D1,B2,C2,D2,Cf]
#    [accX,accY,accRot] = eng.modelDx_pymod(vx,vy,vrot,beta,accRearAxle,tv, param, nargout=3)  #This function only runs in Matlab Session. Shared Matlab session needed to access this function!
    [accX,accY,accRot] = pymodelDx(vx,vy,vrot,beta,accRearAxle,tv, param)   #Marc's Matlab function translated to python
    
    return accX,accY,accRot
    
    
def odeIntegrator (X0, U, simStep, simIncrement):
    setInput(U)
    ts = np.linspace(0,simStep,int(simStep/simIncrement)+1)
    X1 = odeint(odeInt_dx_dt, X0, ts)
    return X1


def odeInt_dx_dt(X,t):
    theta = float(X[3])
    vx = float(X[4])
    vy = float(X[5])
    vrot = float(X[6])

    beta, accRearAxle, tv = getInput(X[0])

    # beta = float(X[7])
    # accRearAxle = float(X[8])
    # tv = float(X[9])

    [accX,accY,accRot] = MATLABvehicleModel(vx,vy,vrot,beta,accRearAxle,tv)
    vxAbs = vx * np.cos(theta) - vy * np.sin(theta)
    vyAbs = vy * np.cos(theta) + vx * np.sin(theta)

    return [1,vxAbs,vyAbs,vrot,accX,accY,accRot,0,0,0]
    # return [1, vxAbs, vyAbs, vrot, accX, accY, accRot]


def odeIntegratorIVP(X0, U, simStep, simIncrement):
    setInput(U)
    t_eval = np.linspace(0, simStep, int(simStep / simIncrement) + 1)

    X1 = solve_ivp(odeIntIVP_dx_dt, [0, simStep], X0, t_eval=t_eval, method='RK45', vectorized=True)

    return np.transpose(X1.y)

def odeIntIVP_dx_dt(t, X):
    theta = float(X[3])
    vx = float(X[4])
    vy = float(X[5])
    vrot = float(X[6])

    beta, accRearAxle, tv = getInput(X[0])

    # beta = float(X[7])
    # accRearAxle = float(X[8])
    # tv = float(X[9])

    #        [accX,accY,accRot] = eng.modelDx_pymod(vx,vy,vrot,beta,accRearAxle,tv, param, nargout=3)
    [accX, accY, accRot] = MATLABvehicleModel(vx, vy, vrot, beta[0], accRearAxle[0], tv[0])
    # [accX, accY, accRot] = MATLABvehicleModel(vx, vy, vrot, beta, accRearAxle, tv)
    vxAbs = vx * np.cos(theta) - vy * np.sin(theta)
    vyAbs = vy * np.cos(theta) + vx * np.sin(theta)
    # return [1, vxAbs, vyAbs, vrot, accX, accY, accRot, 0, 0, 0]
    return [1, vxAbs, vyAbs, vrot, accX, accY, accRot]

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
    [accX,accY,accRot] = MATLABvehicleModel(vx,vy,vrot,beta,accRearAxle,tv)
    vx = vx + accX * simStep
    vy = vy + accY * simStep
    vrot = vrot + accRot * simStep
    x = x + (vx * np.cos(theta) - vy * np.sin(theta)) * simStep
    y = y + (vy * np.cos(theta) + vx * np.sin(theta)) * simStep
    theta = theta + vrot * simStep
    X1 = np.array([[x,y,theta,vx,vy,vrot,beta,accRearAxle,tv]])
    return X1
    
    