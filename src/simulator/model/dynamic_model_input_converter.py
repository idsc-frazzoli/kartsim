#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10.06.19 13:28

@author: mvb
"""
import numpy as np

class MotorFunction:
    def pos_force(self, x, y):
        x2 = x * x
        y2 = y * y
        return -0.321 \
               + 0.1285 * x + 0.002162 * y \
               - 0.03076 * x2 - 0.0002196 * x * y - 3.999e-07 * y2 \
               + 0.002858 * x2 * x + 5.106e-06 * x2 * y + 7.048e-08 * x * y2 - 5.126e-11 * y2 * y


    def neg_force(self, x, y):
        x2 = x * x
        y2 = y * y
        return -0.3738 \
               - 0.06382 * x + 0.002075 * y \
               + 0.03953 * x2 + 0.0001024 * x * y + 1.371e-06 * y2\
               - 0.004336 * x2 * x - 3.495e-06 * x2 * y + 2.634e-08 * x * y2 + 2.899e-10 * y2 * y


    def forward_acceleration(self, forward_vehicle_speed, motor_current):
        current_threshold = 100
        if motor_current > current_threshold:
            return self.pos_force(forward_vehicle_speed, motor_current)
        if motor_current < -current_threshold:
          return self.neg_force(forward_vehicle_speed, motor_current)
        posval = self.pos_force(forward_vehicle_speed, current_threshold)
        negval = self.neg_force(forward_vehicle_speed, -current_threshold)
        prog = (motor_current + current_threshold) / (2 * current_threshold)
        return prog * posval + (1 - prog) * negval


    def backward_acceleration(self, forward_vehicle_speed, motor_current):
        return -self.forward_acceleration(-forward_vehicle_speed, -motor_current)


    def total_acceleration(self, forward_vehicle_speed, motor_current):
        speedthreshold = 0.5
        if forward_vehicle_speed > speedthreshold:
          return self.forward_acceleration(forward_vehicle_speed, motor_current)
        if forward_vehicle_speed < -speedthreshold:
          return self.backward_acceleration(forward_vehicle_speed, motor_current)
        forwardValue = self.forward_acceleration(speedthreshold, motor_current)
        backwardValue = self.backward_acceleration(-speedthreshold, motor_current)
        prog = (forward_vehicle_speed + speedthreshold) / (2 * speedthreshold)
        return (prog * forwardValue + (1 - prog) * backwardValue)


    def get_result(self, X, Y):
        return float(self.total_acceleration(X, Y))


    def get_vectorized_motor_function(self):
        return np.vectorize(self.get_result)


class BrakeFunction:

    def brake_function(self, brake_position):
        if brake_position < 0.025:
            deceleration = 0
        elif brake_position < 0.05:
            deceleration = (brake_position - 0.025) * 255.35
        else:
            deceleration = 6.38375
        return deceleration

    def get_result(self, X):
        return float(self.brake_function(X))

    def get_vectorized_brake_function(self):
        return np.vectorize(self.get_result)


class SteeringFunction:
    def steering_function(self, steer_encoder_val):
        turning_angle = -0.63 * np.power(steer_encoder_val, 3) + 0.94 * steer_encoder_val
        return turning_angle

    def get_result(self, X):
        return float(self.steering_function(X))

    def get_vecortized_steering_function(self):
        return np.vectorize(self.get_result)