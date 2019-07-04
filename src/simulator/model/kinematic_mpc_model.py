#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 28.05.19 09:38

@author: mvb
"""
import numpy as np
from simulator.model.dynamic_model_input_converter import MotorFunction, BrakeFunction, SteeringFunction


class KinematicVehicleMPC:

    def __init__(self, wheel_base=1.19):
        self.name = "kinematic_vehicle_mpc"
        self.wheel_base = wheel_base

        self.motor_function = MotorFunction().get_vectorized_motor_function()
        self.brake_function = BrakeFunction().get_vectorized_brake_function()
        self.steering_function = SteeringFunction().get_vecortized_steering_function()

    def get_name(self):
        return self.name

    def get_state_changes(self, X, U):
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

    def get_accelerations(self, X, U):
        wheel_base = 1.19 # length of vehicle

        velx = X[:, 0]
        vely = X[:, 1]
        velrotz = X[:, 2]
        steering_angle = U[:, 0]
        brake_position = U[:, 1]
        motor_current_l = U[:, 2]
        motor_current_r = U[:, 3]
        dBETA = U[:, 4]

        V = [X[1][0], X[2][0], X[3][0]]
        # V = [X[4][0], X[5][0], X[6][0]]
        # VELX, VELY, VELROTZ = V
        # steering_angle, brake_position, motor_current_l, motor_current_r, dBETA = U

        BETA, AB, TV = self.transform_inputs(steering_angle,
                                             brake_position,
                                             motor_current_l,
                                             motor_current_r,
                                             velx)

        # c, s = np.cos(float(P[2])), np.sin(float(P[2]))
        # R = np.array(((c, -s), (s, c)))
        # Vabs = np.matmul(V[:2], R.transpose())

        # VELROTZ = V[0] / wheel_base * np.tan(BETA)
        dVELROTZ = AB / wheel_base * np.tan(BETA) + V[0] / wheel_base * 1/np.square(np.cos(BETA)) * dBETA

        return np.array([AB, np.zeros((len(AB))), dVELROTZ]).transpose()

    def transform_inputs(self, steering_angle, brake_position, motor_current_l, motor_current_r, velx):
        brake_acceleration = self.brake_function(brake_position)

        acceleration_left_wheel = self.motor_function(velx, motor_current_l) - np.sign(velx) * brake_acceleration
        acceleration_right_wheel = self.motor_function(velx, motor_current_r) - np.sign(velx) * brake_acceleration

        acceleration_rear_axle = (acceleration_left_wheel + acceleration_right_wheel) / 2.0
        torque_tv = (acceleration_right_wheel - acceleration_left_wheel) / 2.0

        turning_angle = -0.63 * np.power(steering_angle, 3) + 0.94 * steering_angle

        return turning_angle, acceleration_rear_axle, torque_tv