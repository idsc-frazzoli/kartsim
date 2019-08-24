#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 28.05.19 09:38

@author: mvb
"""
import numpy as np
from simulator.model.dynamic_model_input_converter import MotorFunction, BrakeFunction, SteeringFunction


class KinematicVehicleMPC:
    def __init__(self, wheel_base=1.19, dist_cog_rear_axle= 0.46, direct_input=True):
        self.name = "mpc_kinematic"
        self.wheel_base = wheel_base
        self.dist_cog_rear_axle = dist_cog_rear_axle
        self.direct_input = direct_input
        self.motor_function = MotorFunction().get_vectorized_motor_function()
        self.brake_function = BrakeFunction().get_vectorized_brake_function()
        self.steering_function = SteeringFunction().get_vecortized_steering_function()

    def get_name(self):
        return self.name

    def get_direct_input_mode(self):
        return self.direct_input

    def get_state_changes(self, X, U):
        wheel_base = 1.19  # length of vehicle
        P = [X[1], X[2], X[3]]
        V = [X[4], 0, X[6]]
        # VELX, VELY, VELROTZ = V
        BETA, AB, TV = U

        c, s = np.cos(float(P[2])), np.sin(float(P[2]))
        R = np.array(((c, -s), (s, c)))
        Vabs = np.matmul(V[:2], R.transpose())

        VELROTZ = V[0] / wheel_base * np.tan(BETA)

        return [Vabs[0], Vabs[1], VELROTZ, AB, 0, 0]

    def get_accelerations(self, velocities, system_inputs):
        if not self.direct_input:
            if isinstance(velocities, list):
                velx, vely, velrotz = velocities
                steering_angle, brake_position, motor_current_l, motor_current_r, turning_rate = system_inputs
            else:
                velx = velocities[:, 0]
                vely = velocities[:, 1]
                velrotz = velocities[:, 2]
                steering_angle = system_inputs[:, 0]
                brake_position = system_inputs[:, 1]
                motor_current_l = system_inputs[:, 2]
                motor_current_r = system_inputs[:, 3]
                turning_rate = system_inputs[:, 4]

            turning_angle, acceleration_rear_axle, torque_tv = self.transform_inputs(steering_angle,
                                                                                     brake_position,
                                                                                     motor_current_l,
                                                                                     motor_current_r,
                                                                                     velx)
        else:
            if isinstance(velocities, list):
                velx, vely, velrotz = velocities
                if isinstance(velx, np.float64) or isinstance(velx, float):
                    turning_angle, acceleration_rear_axle, torque_tv, turning_rate = system_inputs
                    if abs(velx) < 0.25 and acceleration_rear_axle < 0:
                        acceleration_rear_axle *= velx * 4.0
            else:
                velx = velocities[:, 0]
                vely = velocities[:, 1]
                velrotz = velocities[:, 2]
                turning_angle = system_inputs[:, 0]
                acceleration_rear_axle = system_inputs[:, 1]
                torque_tv = system_inputs[:, 2]
                turning_rate = system_inputs[:, 3]

        # velx = X[:, 0]
        # vely = X[:, 1]
        # velrotz = X[:, 2]
        # steering_angle = U[:, 0]
        # brake_position = U[:, 1]
        # motor_current_l = U[:, 2]
        # motor_current_r = U[:, 3]
        # dBETA = U[:, 4]

        # V = [X[1][0], X[2][0], X[3][0]]
        # V = [X[4][0], X[5][0], X[6][0]]
        # VELX, VELY, VELROTZ = V
        # steering_angle, brake_position, motor_current_l, motor_current_r, dBETA = U

        # BETA, AB, TV = self.transform_inputs(steering_angle,
        #                                      brake_position,
        #                                      motor_current_l,
        #                                      motor_current_r,
        #                                      velx)

        # c, s = np.cos(float(P[2])), np.sin(float(P[2]))
        # R = np.array(((c, -s), (s, c)))
        # Vabs = np.matmul(V[:2], R.transpose())

        # VELROTZ = V[0] / wheel_base * np.tan(BETA)
        if turning_angle != 0:
            turn_circle_midpoint_steer = self.wheel_base / np.tan(turning_angle)
        else:
            turn_circle_midpoint_steer = 1000000
        velrotz_target = velx/turn_circle_midpoint_steer
        k = 10
        print(velrotz_target, velrotz)
        # print(turn_circle_midpoint_steer)
        # dVELROTZ = acceleration_rear_axle / self.wheel_base * np.tan(turning_angle) + \
        #            velx / self.wheel_base * 1 / np.square(np.cos(turning_angle)) * turning_rate + k * (velrotz_target - velrotz)
        dVELROTZ = k * (velrotz_target - velrotz)
        if isinstance(acceleration_rear_axle, float):
            return [acceleration_rear_axle, 0, dVELROTZ]
        else:
            return [acceleration_rear_axle, np.zeros((len(acceleration_rear_axle))), dVELROTZ]

    def transform_inputs(self, steering_angle, brake_position, motor_current_l, motor_current_r, velx):
        brake_acceleration_factor = 1
        if isinstance(velx, np.float64) or isinstance(velx, float):
            if abs(velx) < 0.25:
                brake_acceleration_factor = abs(velx) * 4.0
            # pass
        else:
            brake_acceleration_factor = np.add(np.multiply(np.array([abs(vx) < 0.05 for vx in velx]), -0.9), 1)

        brake_acceleration = self.brake_function(brake_position)

        acceleration_left_wheel = self.motor_function(velx, motor_current_l) - np.sign(
            velx) * brake_acceleration * brake_acceleration_factor
        acceleration_right_wheel = self.motor_function(velx, motor_current_r) - np.sign(
            velx) * brake_acceleration * brake_acceleration_factor

        acceleration_rear_axle = (acceleration_left_wheel + acceleration_right_wheel) / 2.0
        torque_tv = (acceleration_right_wheel - acceleration_left_wheel) / 2.0

        turning_angle = -0.63 * np.power(steering_angle, 3) + 0.94 * steering_angle

        return turning_angle, acceleration_rear_axle, torque_tv
