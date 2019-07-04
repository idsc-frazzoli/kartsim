#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.06.19 15:50

@author: mvb
"""
import tensorflow as tf
import numpy as np
from learned_model.ml_model.mlp_keras import MultiLayerPerceptron
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC


class DataDrivenVehicleModel:

    def __init__(self, model_name='FirstTry'):
        self.name = "data_driven_vehicle_model"

        # Load the NN
        self.mlp = MultiLayerPerceptron(model_name=model_name, predict_only=True)
        self.mlp.load_model()
        self.mlp.load_checkpoint('best')
        self.weights, self.biases = self.mlp.get_weights()

        # Load parameters for normalizing inputs
        norm_params = self.mlp.load_normalizing_parameters()
        self.means = norm_params['mean'].values
        self.stds = norm_params['standard deviation'].values

        # Load nominal vehicle model
        self.mpc = DynamicVehicleMPC()

    def get_name(self):
        return self.name

    def normalize_input(self, input):
        return (input - self.means) / self.stds

    def get_accelerations(self, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if isinstance(initial_velocities, list):
            input = np.array([initial_velocities+system_inputs])

            normed_input = self.normalize_input(input)

            disturbance = self.solve_NN(normed_input[0])

            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            # print('disturbance',disturbance)
            # print('accelerations',accelerations)
            result = np.array(accelerations).transpose() + disturbance

            return result[0]
        else:
            input = np.hstack((initial_velocities,system_inputs))

            normed_input = self.normalize_input(input)

            disturbance = self.mlp.predict(normed_input)

            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            # print('disturbance', disturbance)
            # print('accelerations', np.array(accelerations).transpose())
            result = np.array(accelerations).transpose() + disturbance

            return result

    def solve_NN(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            if i < len(self.biases) - 1:
                x = np.maximum(sol, 0)
            else:
                x = sol
        return x
