#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 07.06.19 15:54

@author: mvb
"""
import numpy as np
from simulator.model.dynamic_model_input_converter import MotorFunction, BrakeFunction, SteeringFunction

class DynamicVehicleMPC:
    
    def __init__(self, model_parameters=[9*1.5,1,10,5.2*1.5,1.1,10,0.3], wheel_base=1.19, track_width=1.0, dist_cog_front_axle= 0.73, direct_input=False):
        self.name = "mpc_dynamic"
        self.wheel_base = wheel_base
        self.track_width = track_width # real value: 1.08
        self.dist_cog_front_axle = dist_cog_front_axle
        self.dist_cog_rear_axle = self.wheel_base - dist_cog_front_axle
        self.weight_portion_front = self.dist_cog_rear_axle / self.wheel_base
        self.weight_portion_rear = self.dist_cog_front_axle / self.wheel_base

        self.params_tire_front = model_parameters[:3] #B1, C1, D1
        self.params_tire_rear = model_parameters[3:6] #B2, C2, D2
        self.params_inertia = model_parameters[6]

        self.direct_input = direct_input
        
        self.regularization_factor = 0.5

        self.motor_function = MotorFunction().get_vectorized_motor_function()
        self.brake_function = BrakeFunction().get_vectorized_brake_function()
        self.steering_function = SteeringFunction().get_vecortized_steering_function()

    def get_name(self):
        return self.name

    def get_accelerations(self, velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if not self.direct_input:
            if isinstance(velocities, list):
                velx, vely, velrotz = velocities
                steering_angle, brake_position, motor_current_l, motor_current_r = system_inputs
            else:
                velx = velocities[:, 0]
                vely = velocities[:, 1]
                velrotz = velocities[:, 2]
                steering_angle = system_inputs[:, 0]
                brake_position = system_inputs[:, 1]
                motor_current_l = system_inputs[:, 2]
                motor_current_r = system_inputs[:, 3]

            turning_angle, acceleration_rear_axle, torque_tv = self.transform_inputs(steering_angle,
                                                                                     brake_position,
                                                                                     motor_current_l,
                                                                                     motor_current_r,
                                                                                     velx)
        else:
            if isinstance(velocities, list):
                velx, vely, velrotz = velocities
                turning_angle, acceleration_rear_axle, torque_tv = system_inputs
            else:
                velx = velocities[:, 0]
                vely = velocities[:, 1]
                velrotz = velocities[:, 2]
                turning_angle = system_inputs[:, 0]
                acceleration_rear_axle = system_inputs[:, 1]
                torque_tv = system_inputs[:, 2]

        f1_velx = np.sum(np.multiply(self._rotmat(turning_angle)[0],
                                     np.array([velx, vely + self.dist_cog_front_axle * velrotz])), axis=0)
        f1_vely = np.sum(np.multiply(self._rotmat(turning_angle)[1],
                                     np.array([velx, vely + self.dist_cog_front_axle * velrotz])), axis=0)
        f1y = self._simplefaccy(f1_vely, f1_velx)

        if isinstance(f1y, np.float64):
            F1x = np.sum(np.multiply(self._rotmat(-turning_angle)[0], np.array([np.zeros(1), f1y])),
                         axis=0) * self.weight_portion_front
            F1y = np.sum(np.multiply(self._rotmat(-turning_angle)[1], np.array([np.zeros(1), f1y])),
                         axis=0) * self.weight_portion_front
        else:
            F1x = np.sum(np.multiply(self._rotmat(-turning_angle)[0], np.array([np.zeros(len(f1y)), f1y])),
                         axis=0) * self.weight_portion_front
            F1y = np.sum(np.multiply(self._rotmat(-turning_angle)[1], np.array([np.zeros(len(f1y)), f1y])),
                         axis=0) * self.weight_portion_front

        F2x = acceleration_rear_axle

        F2y1 = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz, velx, (acceleration_rear_axle + torque_tv / 2.) / self.weight_portion_rear) * self.weight_portion_rear / 2.
        F2y2 = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz, velx, (
                    acceleration_rear_axle - torque_tv / 2.) / self.weight_portion_rear) * self.weight_portion_rear / 2.
        F2y = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz, velx,
                               acceleration_rear_axle / self.weight_portion_rear) * self.weight_portion_rear
        TVTrq = torque_tv * self.track_width

        ACCROTZ = (TVTrq + F1y * self.dist_cog_front_axle - F2y * self.dist_cog_rear_axle) / self.params_inertia
        ACCX = F1x + F2x + velrotz * vely
        ACCY = F1y + (F2y1 + F2y2) - velrotz * velx
        # print('f1y :{:5.2f}  velx :{:5.2f}  vely :{:5.2f}  velrotz :{:5.2f}'.format(f1y, velx, vely, velrotz))
        # print('f1y :{:5.2f}  f1_velx :{:5.2f}  f1_vely :{:5.2f}  ACCX :{:5.2f}  ACCY :{:5.2f}  ACCROTZ :{:5.2f}'.format(f1y, f1_velx, f1_vely, ACCX[0], ACCY[0], ACCROTZ[0]))
        # print('F1y:{:5.2f} F2y1 + F2y2:{:5.2f} velrotz*velx:{:5.2f}'.format(F1y[0], F2y1 + F2y2, velrotz * velx))
        # print('F1x:{:5.2f} F2x:{:5.2f} velrotz*vely:{:5.2f}'.format(F1x[0], F2x, velrotz * vely))

        return [ACCX, ACCY, ACCROTZ]

    def transform_inputs(self, steering_angle, brake_position, motor_current_l, motor_current_r, velx):
        brake_acceleration_factor = 1
        if isinstance(velx, np.float64) or isinstance(velx, float):
            if abs(velx) < 0.05:
                brake_acceleration_factor = 0.1
        else:
            brake_acceleration_factor = np.add(np.multiply(np.array([abs(vx) < 0.05 for vx in velx]),-0.9),1)

        brake_acceleration = self.brake_function(brake_position)

        acceleration_left_wheel = self.motor_function(velx, motor_current_l) - np.sign(velx) * brake_acceleration * brake_acceleration_factor
        acceleration_right_wheel = self.motor_function(velx, motor_current_r) - np.sign(velx) * brake_acceleration * brake_acceleration_factor

        acceleration_rear_axle = (acceleration_left_wheel + acceleration_right_wheel) / 2.0
        torque_tv = (acceleration_right_wheel - acceleration_left_wheel) / 2.0

        turning_angle = -0.63 * np.power(steering_angle, 3) + 0.94 * steering_angle

        return turning_angle, acceleration_rear_axle, torque_tv

    def _rotmat(self, beta):
        return np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta), np.cos(beta)]])

    def _magic(self, s, params):
        return params[2] * np.sin(params[1] * np.arctan(params[0] * s))

    def _capfactor(self, taccx):
        if isinstance(taccx, np.float64):
            return (1.0 - self._satfun((taccx / self.params_tire_rear[2]) ** 2.0)) ** (1.0 / 2.0)
        else:
            return (1.0 - self._satfun_approx((taccx / self.params_tire_rear[2]) ** 2.0)) ** (1.0 / 2.0)

    def _simpleslip(self, VELY, VELX, taccx):
        return -(1 / self._capfactor(taccx)) * VELY / (abs(VELX) + self.regularization_factor)

    def _simplediraccy(self, VELY, VELX, taccx):
        return self._magic(self._simpleslip(VELY, VELX, taccx), self.params_tire_rear)

    def _simpleaccy(self, VELY, VELX, taccx):
        return self._capfactor(taccx) * self._simplediraccy(VELY, VELX, taccx)

    def _simplefaccy(self, VELY, VELX):
        # print('magic vy', -VELY , 'magic vx', (abs(VELX) + self.regularization_factor))
        return self._magic(-VELY / (abs(VELX) + self.regularization_factor), self.params_tire_front)

    def _satfun(self, x):
        l = 0.8
        r = 1-l
        if isinstance(x, float):
            if x<l:
                y=x
            elif x<1+r:
                d = (1+r-x)/r
                y = 1-1/4*r*d**2
            else:
                y = 1
            y=0.95*y
        else:
            print('ERROR: x in _satfun(x) is not float!')
        return y

    def _satfun_approx(self, x):
        a = 1.03616764e+00
        b = -1.03619930e+00
        c = -2.91844486e-01
        d = 7.27342214e+00
        e = -1.04920374e+00
        f = 3.59055456e-01
        g = 4.58406776e-01
        h = -3.10005810e-02
        i = 1.70845190e-02
        j = -9.54399532e+01
        k = 6.32433105e-01
        l = -1.97634176e+01
        m = -4.60294363e-02
        return (a * x + b) * (c * np.arctan(d * (x + e) + f) + g) + (h * np.arctan(i * (x + j) + k) + l) * (m)