#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 09:31

@author: mvb
"""
import numpy as np
import time
from simulator.integrate.systeminputhelper import getInputAccel, get_input, get_input_direct, get_kinematic_input, get_kinematic_input_direct

class SystemEquation:
    def __init__(self, vehicle_model):
        self.vehicle_model = vehicle_model
        # print(vehicle_model.get_name())
        self.vehicle_model_name = vehicle_model.get_name()
        self.direct_input = vehicle_model.get_direct_input_mode()
        # if self.vehicle_model_name in ["mpc_kinematic", 'hybrid_kinematic_mlp']:
        #     if self.direct_input:
        #         self.get_input = get_kinematic_input_direct
        #     else:
        #         self.get_input = get_kinematic_input
        # else:
        if self.direct_input:
            self.get_input = get_input_direct
        else:
            self.get_input = get_input

    # def initialize_vehicle_model(model_object):
    #     global vehicle_model
    #     vehicle_model = model_object
    #     print('initialized')

    def get_system_equation(self):
        if self.vehicle_model_name in ["mpc_dynamic", "hybrid_mlp", "mlp", "no_model"]:
            return self.solveivp_dynamic_dx_dt
        elif self.vehicle_model_name in ["hybrid_lstm"]:
            return self.solveivp_hybrid_lstm_dx_dt
        elif self.vehicle_model_name in ["mpc_kinematic", "hybrid_kinematic_mlp"]:
            return self.solveivp_kinematic_dx_dt
        elif self.vehicle_model_name == "acceleration_reference_model":
            return self.solveivp_reference_dx_dt

    def get_vehicle_model_name(self):
        return self.vehicle_model_name

    def odeint_dx_dt(self, X, t):
        V = X[4:7]
        U = self.get_input(X[0])

        V_dt = self.vehicle_model.get_accelerations(V, U)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_dynamic_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        U = self.get_input(X[0])
        V_dt = self.vehicle_model.get_accelerations(V, U)
        # V_dt_a, V_dt_d = self.vehicle_model.get_accelerations(V, U)
        # V_dt = V_dt_a + V_dt_d
        # V_dt = V_dt[0]
        # if X[0] > 0.0:
        #     print(
        #         't,{:5.4f}, V_dt,{:5.4f}, {:5.4f}, {:5.4f}, V,{:5.4f}, {:5.4f}, {:5.4f}, U,{:5.4f}, {:5.4f}, {:5.4f}'.format(
        #             X[0],
        #             # V_dt[0][0],
        #             # V_dt[1][0],
        #             # V_dt[2][0],
        #             V_dt[0],
        #             V_dt[1],
        #             V_dt[2],
        #             V[0],
        #             V[1],
        #             V[2],
        #             U[0],
        #             U[1],
        #             U[2],
                    # U[3],
                    # V_dt_d[0],
                    # V_dt_d[1],
                    # V_dt_d[2],
                    # V_dt_a[0][0],
                    # V_dt_a[0][1],
                    # V_dt_a[0][2],
                # ))

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_hybrid_lstm_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        U = self.get_input(X[0])

        V_dt = self.vehicle_model.get_accelerations(X[0], V, U)

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def solveivp_kinematic_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        U = self.get_input(X[0])
        V_dt = self.vehicle_model.get_accelerations(V, U)

        # if X[0] >= 0.0:
        #     print(
        #         't,{:5.4f}, V_dt,{:5.4f}, {:5.4f}, {:5.4f}, V,{:5.4f}, {:5.4f}, {:5.4f}, U,{:5.4f}, {:5.4f}, {:5.4f}, {:5.4f}'.format(
        #             X[0],
        #             V_dt[0],
        #             V_dt[1],
        #             V_dt[2],
        #             V[0],
        #             V[1],
        #             V[2],
        #             U[0],
        #             U[1],
        #             U[2],
        #             U[3],
        #             # U[4],
        #         ))

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())
        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]

    def euler_dx_dt(self, V, U):
        return self.vehicle_model.get_accelerations(V, U)

    def solveivp_reference_dx_dt(self, t, X):
        V = [X[4], X[5], X[6]]
        V_dt = getInputAccel(X[0])

        c, s = np.cos(float(X[3])), np.sin(float(X[3]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        return [1, Vabs[0], Vabs[1], V[2], V_dt[0], V_dt[1], V_dt[2]]
