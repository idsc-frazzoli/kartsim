#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.06.19 15:50

@author: mvb
"""
import tensorflow as tf
import numpy as np
from learned_model.ml_models.mlp_keras import MultiLayerPerceptron as MultiLayerPerceptronNormal
from learned_model_for_mpc.ml_models.mlp_keras import MultiLayerPerceptron as MultiLayerPerceptronMPCInput
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC


class DataDrivenVehicleModel:

    def __init__(self, model_name='FirstTry', direct_input=False):
        self.name = "hybrid_mlp"
        self.direct_input = direct_input
        # Load the NN
        if self.direct_input:
            self.mlp = MultiLayerPerceptronMPCInput(model_name=model_name, predict_only=True)
        else:
            self.mlp = MultiLayerPerceptronNormal(model_name=model_name, predict_only=True)
        self.mlp.load_model()
        self.mlp.load_checkpoint('best')
        self.weights, self.biases = self.mlp.get_weights()

        if 'relu' in model_name:
            self.disturbance = self.solve_NN_relu
        elif 'softplus' in model_name:
            self.disturbance = self.solve_NN_softplus

        # Load parameters for normalizing inputs
        norm_params = self.mlp.load_normalizing_parameters()
        self.means = norm_params['mean'].values
        self.stds = norm_params['standard deviation'].values

        # Load nominal vehicle model
        self.mpc = DynamicVehicleMPC(direct_input=self.direct_input)

    def get_name(self):
        return self.name

    def get_direct_input_mode(self):
        return self.direct_input

    def normalize_input(self, input):
        return (input - self.means) / self.stds

    def get_accelerations(self, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if isinstance(initial_velocities, list):
            features = np.array([initial_velocities+system_inputs])

            if 'sym' in self.model_name:
                features = self.mlp.symmetry_dim_reduction(features)
            normalized_features = self.normalize_input(features)

            disturbance = self.disturbance(normalized_features[0])

            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            # print(f'disturbance {disturbance}   acceleration {accelerations}')
            result = np.array(accelerations).transpose() + disturbance
            return result[0]
        else:
            input = np.hstack((initial_velocities,system_inputs))

            disturbance = self.mlp.predict(input)

            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            # print('disturbance', disturbance)
            # print('accelerations', np.array(accelerations).transpose())
            result = np.array(accelerations).transpose() + disturbance

            return result

    def solve_NN_relu(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            if i < len(self.biases) - 1:
                x = np.maximum(sol, 0)
            else:
                x = sol
        return x

    def solve_NN_softplus(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            if i < len(self.biases) - 1:
                x = np.log(np.exp(sol) + 1)
            else:
                x = sol
        return x
