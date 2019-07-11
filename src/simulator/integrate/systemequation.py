#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:31

@author: mvb
"""
import numpy as np
import time
from simulator.integrate.systeminputhelper import getInput, getInputAccel


class SystemEquation:
    def __init__(self, vehicle_model):
        self.vehicle_model = vehicle_model
        self.vehicle_model_name = vehicle_model.get_name()

    # def initialize_vehicle_model(model_object):
    #     global vehicle_model
    #     vehicle_model = model_object
    #     print('initialized')

    def get_system_equation(self):
        if self.vehicle_model_name in ["mpc_dynamic", "hybrid_mlp"]:
            return self.solveivp_dynamic_dx_dt
        elif self.vehicle_model_name in ["hybrid_lstm"]:
            return self.solveivp_hybrid_lstm_dx_dt
        elif self.vehicle_model_name == "kinematic_vehicle_mpc":
            return self.solveivp_kinematic_dx_dt
        elif self.vehicle_model_name == "acceleration_reference_model":
            return self.solveivp_reference_dx_dt

    def get_vehicle_model_name(self):
        return self.vehicle_model_name

    def odeint_dx_dt(self, X, t):
        V = X[4:7]
        U = getInput(X[0])

        V_dt = self.vehicle_model.get_accelerations(V, U)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_dynamic_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        U = getInput(X[0])

        V_dt = self.vehicle_model.get_accelerations(V, U)

        # if t > 0.999:
        #     print('_______')
        # if t < 0.5:
        #     # print('t:{:5.4f}'.format(t))
        #     print('t:{:5.4f}, V_dt:[{:5.2f} {:5.2f} {:5.2f}], V:[{:5.4f} {:5.4f} {:5.4f}], U:[{:5.4f} {:5.4f} {:5.2f} {:5.2f}]'.format(t, V_dt[0][0],
        #                                                                                           V_dt[1][0],
        #                                                                                           V_dt[2][0],
        #                                                                                           V[0],
        #                                                                                           V[1],
        #                                                                                           V[2],
        #                                                                                           U[0],
        #                                                                                           U[1],
        #                                                                                           U[2],
        #                                                                                           U[3],
        #                                                                                           ))

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_hybrid_lstm_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        U = getInput(X[0])

        V_dt = self.vehicle_model.get_accelerations(X[0], V, U)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_kinematic_dx_dt(self, t, X):

        U = getInput(X[0])

        X_dt = self.vehicle_model.get_state_changes(X, U)

        return [1, X_dt[0], X_dt[1], X_dt[2], X_dt[3], X_dt[4], X_dt[5]]

    def euler_dx_dt(self, V, U):
        return self.vehicle_model.get_accelerations(V, U)

    def solveivp_reference_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        V_dt = getInputAccel(X[0])

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]
