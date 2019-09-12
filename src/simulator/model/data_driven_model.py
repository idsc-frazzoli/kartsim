#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.06.19 15:50

@author: mvb
"""
import tensorflow as tf
import numpy as np
from learned_model.ml_models.mlp_keras import MultiLayerPerceptron
from learned_model_for_mpc.ml_models.mlp_keras_mpc import MultiLayerPerceptronMPC
from learned_model_for_mpc.ml_models.mlp_keras_mpc_additfeatures import MultiLayerPerceptronMPCAdditFeatures
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC


class DataDrivenVehicleModel:

    def __init__(self, model_type='hybrid_mlp', model_name='FirstTry', direct_input=False):
        self.model_type = model_type
        self.model_name = model_name
        self.direct_input = direct_input
        # Load the NN
        if self.direct_input:
            if 'squared' in self.model_name:
                self.mlp = MultiLayerPerceptronMPCAdditFeatures(model_name=model_name, predict_only=True)
            else:
                self.mlp = MultiLayerPerceptronMPC(model_name=model_name, predict_only=True)
        else:
            self.mlp = MultiLayerPerceptron(model_name=model_name, predict_only=True)
        self.mlp.load_model()
        self.mlp.load_checkpoint('best')
        self.weights, self.biases = self.mlp.get_weights()
        if 'relu' in model_name:
            self.disturbance = self.solve_NN_relu
        elif 'softplus' in model_name:
            self.disturbance = self.solve_NN_softplus
        elif 'tanh' in model_name:
            self.disturbance = self.solve_NN_tanh
        elif 'None' in model_name:
            self.disturbance = self.solve_NN_None

        # Load parameters for normalizing inputs
        norm_params = self.mlp.load_normalizing_parameters()
        self.means = norm_params['mean'].values
        self.stds = norm_params['standard deviation'].values

        # Load nominal vehicle model
        if self.model_type == 'hybrid_mlp':
            self.nominal_model = DynamicVehicleMPC(direct_input=self.direct_input)
        elif self.model_type == 'hybrid_kinematic_mlp':
            self.nominal_model = KinematicVehicleMPC(direct_input=self.direct_input)
        elif self.model_type == 'no_model':
            self.nominal_model = None
        else:
            raise ValueError('No or invalid vehicle model type. Please specify.')


    def get_name(self):
        return self.model_type

    def get_direct_input_mode(self):
        return self.direct_input

    def normalize_input(self, input):
        return (input - self.means) / self.stds

    def get_accelerations(self, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if isinstance(initial_velocities, list):
            features = np.array([initial_velocities+system_inputs])
            if 'sym' in self.model_name:
                features, label_mirror = self.mlp.mirror_state_space(features)
                if 'squared' in self.model_name:
                    features = self.mlp.add_features(features)
                normalized_features = self.normalize_input(features)
                disturbance = self.disturbance(normalized_features[0])
                disturbance[1:3] = disturbance[1:3] * label_mirror[0]
            else:
                if 'squared' in self.model_name:
                    features = self.mlp.add_features(features)
                normalized_features = self.normalize_input(features)
                disturbance = self.disturbance(normalized_features[0])

            if self.nominal_model is not None:
                accelerations = self.nominal_model.get_accelerations(initial_velocities, system_inputs)
            else:
                accelerations = [0,0,0]
            result = np.array(accelerations) + disturbance
            return result
        else:
            input = np.hstack((initial_velocities,system_inputs))

            disturbance = self.mlp.predict(input)

            if self.nominal_model is not None:
                accelerations = self.nominal_model.get_accelerations(initial_velocities, system_inputs)
            else:
                accelerations = np.array([0,0,0])
            # print('disturbance', disturbance)
            # print('accelerations', np.array(accelerations).transpose())
            result = accelerations + disturbance

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

    def solve_NN_tanh(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            if i < len(self.biases) - 1:
                x = np.tanh(sol)
            else:
                x = sol
        return x

    def solve_NN_None(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            x = sol
        return x
