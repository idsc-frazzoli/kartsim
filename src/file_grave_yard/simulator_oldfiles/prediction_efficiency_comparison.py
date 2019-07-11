#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 17.06.19 14:14

@author: mvb
"""
import tensorflow as tf
import numpy as np
from learned_model.ml_models.mlp_keras import MultiLayerPerceptron
from simulator.model.data_driven_model import DataDrivenVehicleModel

import time

vehicle_model = DataDrivenVehicleModel()
# # [[-0.95633353  0.04260441 -0.30555203 -0.05212023 -0.66840983  0.38073336
# #    0.15132715]]
#
V=[0.2,0.0,0.0]
U=[0.2,0.0,500,500]
#
# V=[-0.95633353,  0.04260441, -0.30555203]
# U=[-0.05212023, -0.66840983,  0.38073336,0.15132715]
for i in range(10):
    t0 = time.time()
    acc = vehicle_model.get_accelerations(V,U)
    print('t class', time.time()-t0)

# #Load the NN
# mlp = MultiLayerPerceptron(model_name='FirstTry')
# mlp.load_model()
# mlp.load_checkpoint('best')
# W,B = mlp.get_weights()
#
# # Convert NN to function
#
# def relu(x):
#     return np.maximum(x, 0)
#
# def layer(inputs,weights,biases):
#     # print(np.sum(weights[:,0]*inputs)+biases[0])
#     # print(weights.transpose().shape)
#     # print(inputs.shape)
#     sol = np.matmul(weights.transpose(),inputs) + biases
#     output = relu(sol)
#     return output
#
# def last_layer(inputs,weights,biases):
#     # print(np.sum(weights[:,0]*inputs)+biases[0])
#     # print(weights.transpose().shape)
#     # print(inputs.shape)
#     sol = np.matmul(weights.transpose(),inputs) + biases
#     # print('bias',biases)
#     output = sol
#     return output
# t0=time.time()
# inputs_0 = vehicle_model.normalize_input(np.array(V+U))
# print('inputs_0',inputs_0)
# inputs_1 = layer(inputs_0, W[0], B[0])
# # print(inputs_1)
# inputs_2 = layer(inputs_1,W[1],B[1])
# # print(inputs_2)
# output = last_layer(inputs_2,W[2],B[2])
# print('t naive',time.time()-t0)
# print(output)
#
# def solve_NN(inputs,weights,biases):
#     input = inputs
#     for i, (w, b) in enumerate(zip(weights,biases)):
#         sol = np.matmul(w.transpose(), input) + b
#         if i < len(biases)-1:
#             input = relu(sol)
#         else:
#             input = sol
#     return input
#
# t0=time.time()
# output1 = solve_NN(inputs_0, W, B)
# print('t united',time.time()-t0)
# print(output1)

#
# # Load parameters for normalizing inputs
# norm_params = mlp.load_normalizing_parameters()
# means = norm_params['mean'].values
# stds = norm_params['standard deviation'].values
#
# def normalize_input(input, means, stds):
#     return (input - means) / stds
#
#
# t0 = time.time()
#
# # [['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
# # 'steer position cal [n.a.]', 'brake position effective [m]',
# # 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
#
# # input = np.array([[5.3,0.5,-0.5,-0.2,0.0,1500,500]])
# input = np.array([[5.3, 0.0, 0.0, 0.0, 0.0, 0, 0]])
#
# normed_input = normalize_input(input, means, stds)
#
# t0 = time.time()
# print(mlp.predict(normed_input))
# print('t direct', time.time()-t0)
#
# print(time.time() - t0)
