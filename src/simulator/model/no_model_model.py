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
from learned_model_for_mpc.ml_models.mlp_keras_mpc_symmetric import MultiLayerPerceptronMPCSymmetric
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC


class NoModelModel:
    def __init__(self, model_type='mlp', model_name='FirstTry', direct_input=False):
        self.model_type = model_type
        self.model_name = model_name
        self.direct_input = direct_input
        # Load the NN
        if self.direct_input:
            if 'squared' in self.model_name:
                self.mlp = MultiLayerPerceptronMPCAdditFeatures(model_name=model_name, predict_only=True)
            elif 'symmetric' in self.model_name:
                self.mlp = MultiLayerPerceptronMPCSymmetric(model_name=model_name, predict_only=True)
            else:
                self.mlp = MultiLayerPerceptronMPC(model_name=model_name, predict_only=True)
        else:
            self.mlp = MultiLayerPerceptron(model_name=model_name, predict_only=True)
        self.mlp.load_model()
        self.mlp.load_checkpoint('best')
        self.weights, self.biases = self.mlp.get_weights()
        if 'symmetric' in model_name:
            if 'tanh' in model_name:
                self.predictor = self.solve_NN_tanh_sym
            elif 'softplus' in model_name:
                self.predictor = self.solve_NN_softplus_sym
            elif 'sigmoid' in model_name:
                self.predictor = self.solve_NN_sigmoid_sym
        else:
            if 'relu' in model_name:
                self.predictor = self.solve_NN_relu
            elif 'softplus' in model_name:
                self.predictor = self.solve_NN_softplus
            elif 'tanh' in model_name:
                self.predictor = self.solve_NN_tanh
            elif 'sigmoid' in model_name:
                self.predictor = self.solve_NN_sigmoid

        # Load parameters for normalizing inputs
        norm_params = self.mlp.load_normalizing_parameters()
        self.means = norm_params['mean'].values
        self.stds = norm_params['standard deviation'].values

    def get_name(self):
        return self.model_type

    def get_direct_input_mode(self):
        return self.direct_input

    def normalize_input(self, input):
        return (input - self.means) / self.stds

    def normalize_input_sym(self, input):
        input[:,0] = input[:,0] - self.means[0]
        input[:,4] = input[:,4] - self.means[4]
        return input / self.stds

    def get_accelerations(self, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if isinstance(initial_velocities, list):
            features = np.array([initial_velocities + system_inputs])
            if 'customfeat' in self.model_name:
                new_features = self.get_custom_feat(features)
                features = np.hstack((features, new_features))
            if 'symmetric' in self.model_name:
                normalized_features = self.normalize_input_sym(features)
            else:
                normalized_features = self.normalize_input(features)
            prediction = self.predictor(normalized_features[0])
            input = np.hstack((np.array(initial_velocities,), np.array(system_inputs,)))
            input = np.array([input])

            result = prediction

            return result
        else:
            input = np.hstack((initial_velocities, system_inputs))
            if 'customfeat' in self.model_name:
                new_features = self.get_custom_feat(input)
                input = np.hstack((input, new_features))
            prediction = self.mlp.predict(input)
            result = prediction
            return result

    def get_custom_feat(self, features):
        features = features.T
        # VY * VTHETA
        new_features = features[1, :] * features[2, :]
        # VTHETA * TV
        new_features = np.vstack((new_features, features[2, :] * features[5, :]))
        # VX * BETA * BETA
        new_features = np.vstack((new_features, features[0, :] * features[3, :] * features[3, :]))
        # VX * VTHETA
        new_features = np.vstack((new_features, features[0, :] * features[2, :]))
        # cos(VX) * VY
        new_features = np.vstack((new_features, np.cos(features[0, :] / 10.0 * np.pi / 2.0) * features[1, :]))
        # VX * BETA
        new_features = np.vstack((new_features, features[0, :] * features[3, :]))
        # VX * TV
        new_features = np.vstack((new_features, features[0, :] * features[5, :]))
        # VX * VTHETA * AB
        new_features = np.vstack((new_features, features[0, :] * features[2, :] * features[4, :]))
        # VX * BETA * AB * AB
        new_features = np.vstack(
            (new_features, features[0, :] * features[3, :] * features[4, :] * features[4, :]))
        # cos(VX) * VTHETA
        new_features = np.vstack((new_features, np.cos(features[0, :] / 10.0 * np.pi / 2.0) * features[2, :]))
        # cos(BETA) * VTHETA
        new_features = np.vstack((new_features, np.cos(features[3, :] / 0.44 * np.pi / 2.0) * features[2, :]))
        return new_features.T

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

    def solve_NN_sigmoid(self, inputs):
        x = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            sol = np.matmul(w.transpose(), x) + b
            if i < len(self.biases) - 1:
                x = 0.5 * (1 + np.tanh(sol/2.0))
            else:
                x = sol
        return x

    def solve_NN_tanh_sym(self, inputs):
        x = inputs
        x_sym = x * np.array([1,-1,-1,-1,1,-1])
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i < len(self.biases) - 2:
                sol = np.matmul(w.transpose(), x) + b
                sol_sym = np.matmul(w.transpose(), x_sym) + b
                x = np.tanh(sol)
                x_sym = np.tanh(sol_sym)
            elif i < len(self.biases) - 1:
                x_all = (x + x_sym) / 2.0
                x_all = np.matmul(w.transpose(), x_all) + b
            else:
                x_odd = (x - x_sym) / 2.0
                x_odd = np.matmul(w.transpose(), x_odd)
                x_all = np.hstack((x_all,x_odd))

        return x_all

    def solve_NN_softplus_sym(self, inputs):
        x = inputs
        x_sym = x * np.array([1,-1,-1,-1,1,-1])
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i < len(self.biases) - 2:
                sol = np.matmul(w.transpose(), x) + b
                sol_sym = np.matmul(w.transpose(), x_sym) + b
                x = np.log(np.exp(sol) + 1)
                x_sym = np.log(np.exp(sol_sym) + 1)
            elif i < len(self.biases) - 1:
                x_all = (x + x_sym) / 2.0
                x_all = np.matmul(w.transpose(), x_all) + b
            else:
                x_odd = (x - x_sym) / 2.0
                x_odd = np.matmul(w.transpose(), x_odd)
                x_all = np.hstack((x_all,x_odd))

        return x_all

    def solve_NN_sigmoid_sym(self, inputs):
        x = inputs
        x_sym = x * np.array([1,-1,-1,-1,1,-1])
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i < len(self.biases) - 2:
                sol = np.matmul(w.transpose(), x) + b
                sol_sym = np.matmul(w.transpose(), x_sym) + b
                x = 0.5 * (1 + np.tanh(sol/2.0))
                x_sym = 0.5 * (1 + np.tanh(sol_sym/2.0))
            elif i < len(self.biases) - 1:
                x_all = (x + x_sym) / 2.0
                x_all = np.matmul(w.transpose(), x_all) + b
            else:
                x_odd = (x - x_sym) / 2.0
                x_odd = np.matmul(w.transpose(), x_odd)
                x_all = np.hstack((x_all,x_odd))

        return x_all
