#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:25 2019

@author: mvb
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp

from simulator.integrate.systemequation import odeint_dx_dt, solveivp_dynamic_dx_dt, solveivp_kinematic_dx_dt
from simulator.integrate.systeminputhelper import setInput
from model.pymodelDx import mpc_dynamic_vehicle_model

def odeIntegrator (X0, U, simStep, simIncrement):
    setInput(U)
    ts = np.linspace(0,simStep,int(simStep/simIncrement)+1)
    X1 = odeint(odeint_dx_dt, X0, ts)
    return X1


def odeIntegratorIVP(X0, U, simStep, simIncrement):
    setInput(U)
    t_eval = np.linspace(0, simStep, int(simStep / simIncrement) + 1)

    # Dynamic mpc model
    # X1 = solve_ivp(solveivp_dynamic_dx_dt, [0, simStep], X0, t_eval=t_eval, method='RK45', vectorized=True, first_step=None,
    #                rtol=1.49012e-8, atol=1.49012e-8)
    X1 = solve_ivp(solveivp_dynamic_dx_dt, [0, simStep], X0, t_eval=t_eval, method='RK45')
    # X1 = solve_ivp(solveivp_dynamic_dx_dt, [0, simStep], X0, t_eval=t_eval, method='LSODA', rtol=1e-8)

    # Kinematic mpc model
    # X0[5] = X0[6] = 0
    # X1 = solve_ivp(solveivp_kinematic_dx_dt, [0, simStep], X0, t_eval=t_eval, method='RK45', vectorized=True)

    return np.transpose(X1.y)


def euler(X0, U, simStep, simIncrement):
    X = X0
    X1 = X
    U = U[1:]
    print('U',len(U[0,:]))
    for i in range(len(U[0,:])):
        Ui = U[:,i]
        V = [X[4], X[5], X[6]]

    #    [accX,accY,accRot] = eng.modelDx_pymod(vx,vy,vrot,beta,accRearAxle,tv, param, nargout=3) #This function only runs in Matlab Session. Shared Matlab session needed to access this function!
        V_dt = mpc_dynamic_vehicle_model(V, Ui)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())
        dX = [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]
        X = X + np.multiply(dX,simIncrement)
        X1 = np.vstack((X1,X))
    print(X1)
    print(type(X1))
    return X1
    
    