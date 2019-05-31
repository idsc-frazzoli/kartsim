#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 28.05.19 09:38

@author: mvb
"""
import numpy as np

def mpc_kinematic_vehicle_model(X, U):
    l = 1.19 # length of vehicle

    P = [X[1][0], X[2][0], X[3][0]]
    V = [X[4][0], X[5][0], X[6][0]]
    # VELX, VELY, VELROTZ = V
    BETA, AB, TV = U

    c, s = np.cos(float(P[2])), np.sin(float(P[2]))
    R = np.array(((c, -s), (s, c)))
    Vabs = np.matmul(V[:2], R.transpose())

    VELROTZ_new = V[0] / l * np.tan(BETA)

    return [Vabs[0], Vabs[1], VELROTZ_new, AB, 0, 0]