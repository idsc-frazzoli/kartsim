#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 28.05.19 09:38

@author: mvb
"""
import numpy as np

class KinematicVehicleMPC:

    def __init__(self, wheel_base=1.19):
        self.name = "kinematic_vehicle_mpc"
        self.wheel_base = wheel_base

    def get_name(self):
        return self.name

    def get_state_changes(X, U):
        wheel_base = 1.19 # length of vehicle

        P = [X[1][0], X[2][0], X[3][0]]
        V = [X[4][0], X[5][0], X[6][0]]
        # VELX, VELY, VELROTZ = V
        BETA, AB, TV = U

        c, s = np.cos(float(P[2])), np.sin(float(P[2]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        VELROTZ = V[0] / wheel_base * np.tan(BETA)

        return [Vabs[0], Vabs[1], VELROTZ, AB, 0, 0]