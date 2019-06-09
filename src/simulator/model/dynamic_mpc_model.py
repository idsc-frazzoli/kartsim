#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 07.06.19 15:54

@author: mvb
"""
import numpy as np

class DynamicVehicleMPC:
    
    def __init__(self, model_parameters=[9,1,10,5.2,1.1,10,0.3], wheel_base=1.19, track_width=1.0, dist_cog_front_axle= 0.73):
        self.name = "dynamic_vehicle_mpc"
        self.wheel_base = wheel_base
        self.track_width = track_width # real value: 1.08
        self.dist_cog_front_axle = dist_cog_front_axle
        self.dist_cog_rear_axle = self.wheel_base - dist_cog_front_axle
        self.weight_portion_front = self.dist_cog_rear_axle / self.wheel_base
        self.weight_portion_rear = self.dist_cog_front_axle / self.wheel_base

        self.params_tire_front = model_parameters[:3] #B1, C1, D1
        self.params_tire_rear = model_parameters[3:6] #B2, C2, D2
        self.params_intetia = model_parameters[6]
        
        self.regularization_factor = 0.5

    def get_name(self):
        return self.name

    def get_accelerations(self, initial_velocities=[0,0,0], system_inputs=[0,0,0,0]):
        velx, vely, velrotz = initial_velocities
        turning_angle, acceleration_rear_axle, torque_tv = system_inputs

        # steering_angle, brake_position, motor_current_l, motor_current_r = system_inputs
        # turning_angle, acceleration_rear_axle, torque_tv = self.transform_inputs(steering_angle,
        #                                                                          brake_position,
        #                                                                          motor_current_l,
        #                                                                          motor_current_r)

        vel1 = np.matmul(self._rotmat(turning_angle), np.array([[velx], [vely + self.dist_cog_front_axle * velrotz]]))
        f1y = self._simplefaccy(vel1[1], vel1[0])

        if isinstance(f1y, np.ndarray):
            f1y = f1y[0]
        F1 = np.matmul(self._rotmat(-turning_angle), np.array([[0], [f1y]])) * self.weight_portion_front
        F1x = F1[0]
        F1y = F1[1]

        F2x = acceleration_rear_axle

        F2y1 = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz,
                                velx,
                                (acceleration_rear_axle + torque_tv / 2.) / self.weight_portion_rear) \
               * self.weight_portion_rear / 2.

        F2y2 = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz,
                                velx,
                                (acceleration_rear_axle - torque_tv / 2.) / self.weight_portion_rear) \
               * self.weight_portion_rear / 2.

        F2y = self._simpleaccy(vely - self.dist_cog_rear_axle * velrotz,
                               velx,
                               acceleration_rear_axle / self.weight_portion_rear) \
              * self.weight_portion_rear

        # if abs(velx) < 0.05 and abs(vely) < 0.05:
        #     torque_tv = 0.
        TVTrq = torque_tv * self.track_width

        ACCROTZ = (TVTrq + F1y * self.dist_cog_front_axle - F2y * self.dist_cog_rear_axle) / self.params_intetia
        ACCX = F1x + F2x + velrotz * vely
        ACCY = F1y + F2y1 + F2y2 - velrotz * velx

        return ACCX[0],ACCY[0],ACCROTZ[0]

    def transform_inputs(self, steering_angle, brake_position, motor_current_l, motor_current_r):

        return 0

    def _rotmat(self, beta):
        return np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta), np.cos(beta)]])

    def _magic(self, s, params):
        return params[2] * np.sin(params[1] * np.arctan(params[0] * s))


    def _capfactor(self, taccx):
        return (1.0 - self._satfun((taccx / self.params_tire_rear[2]) ** 2.0)) ** (1.0 / 2.0)


    def _simpleslip(self, VELY, VELX, taccx):
        # if abs(VELX) < 0.05 and abs(VELY) < 0.05:
        #     VELY = 0.
        #     VELX = 1.
        return -(1 / self._capfactor(taccx)) * VELY / (abs(VELX) + self.regularization_factor)


    def _simplediraccy(self, VELY, VELX, taccx):
        return self._magic(self._simpleslip(VELY, VELX, taccx), self.params_tire_rear)


    def _simpleaccy(self, VELY, VELX, taccx):
        return self._capfactor(taccx) * self._simplediraccy(VELY, VELX, taccx)


    def _simplefaccy(self, VELY, VELX):
        # if abs(VELX) < 0.05 and abs(VELY) < 0.05:
        #     VELY = 0.
        #     VELX = self.regularization_factor + 1.
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

# def satfun_approx(x):
#     a = 1.03616764e+00
#     b = -1.03619930e+00
#     c = -2.91844486e-01
#     d = 7.27342214e+00
#     e = -1.04920374e+00
#     f = 3.59055456e-01
#     g = 4.58406776e-01
#     h = -3.10005810e-02
#     i = 1.70845190e-02
#     j = -9.54399532e+01
#     k = 6.32433105e-01
#     l = -1.97634176e+01
#     m = -4.60294363e-02
#     return (a * x + b) * (c * np.arctan(d * (x + e) + f) + g) + (h * np.arctan(i * (x + j) + k) + l) * (m)