#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:25 2019

@author: mvb
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp

from simulator.integrate.systeminputhelper import set_input, set_input_direct, set_kinematic_input, set_kinematic_input_direct


def odeIntegrator (X0, U, simStep, simIncrement, system_equation=None):
    set_input(U)
    ts = np.linspace(0,simStep,int(simStep/simIncrement)+1)
    X1 = odeint(system_equation.odeint_dx_dt, X0, ts)
    return X1


def odeIntegratorIVP(X0, U, simStep, simIncrement, system_equation=None):
    if system_equation.get_vehicle_model_name() == "mpc_kinematic":
        if len(U) == 4:
            set_kinematic_input_direct(U)
        else:
            set_kinematic_input(U)
    else:
        if len(U) == 4:
            set_input_direct(U)
        else:
            set_input(U)
    t_eval = np.linspace(0, simStep, round(simStep / simIncrement,4) + 1)

    X1 = solve_ivp(system_equation.get_system_equation(), [0, simStep], X0, t_eval=t_eval, method='RK45', rtol=1e-5)
    if X1.status == 0:
        return np.transpose(X1.y)
    else:
        return 'Integration step failed. Kill simulation.'


def euler(X0, U, simIncrement, system_equation=None):
    X = X0
    X1 = X
    U = U[1:]
    for i in range(len(U[0,:])):
        Ui = U[:,i]
        V = [X[4], X[5], X[6]]

    #    [accX,accY,accRot] = eng.modelDx_pymod(vx,vy,vrot,beta,accRearAxle,tv, param, nargout=3) #This function only runs in Matlab Session. Shared Matlab session needed to access this function!
        V_dt = system_equation.euler_dx_dt(V, Ui)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())
        dX = [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]
        X = X + np.multiply(dX,simIncrement)
        X1 = np.vstack((X1,X))
    return X1
