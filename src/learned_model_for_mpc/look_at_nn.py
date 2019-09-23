#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 21.06.19 15:06

@author: mvb
"""

# model_names = ['2x32relu_regularized_more','2x32relu_regularized']
#
# for model_name in model_names:
#     mlp = MultiLayerPerceptron(model_name=model_name, predict_only=True)
#     mlp.load_model()
#     mlp.load_checkpoint('best')
#     weights, biases = mlp.get_weights()
#
#     for i,weight in enumerate(weights):
#         w = np.reshape(weight,(-1,1))
#         plt.figure(model_name + str(i))
#         plt.hist(w,50)
#     plt.show()


from data_visualization.data_io import getPKL
from learned_model_for_mpc.ml_models.mlp_keras_mpc import MultiLayerPerceptronMPC
from learned_model_for_mpc.ml_models.mlp_keras_mpc_symmetric import MultiLayerPerceptronMPCSymmetric
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import os
import config

# model_name = '2x16_softplus_reg0p1_directinput' # mpc dyn + NN
# model_name = '3x16_softplus_reg0p0001_kin_directinput' # kin + NN
# model_name = '2x16_tanh_reg0p0_directinput' #new
# model_name = '0x6_None_reg0p0001_directinput'
# model_name = '0x6_None_reg0p01_directinput'
# model_name = '0x6_None_reg0p1_directinput'
# model_name = '2x16_softplus_reg0p1_directinput'
# model_name = '2x16_softplus_reg0p01_directinput'
# model_name = '1x16_tanh_reg0p0_directinput'
# model_name = '1x16_softplus_reg0p01_directinput'
# model_name = '2x16_tanh_reg0p0001_kin_directinput'
# model_name = '2x16_softplus_reg0p0001_kin_directinput'
# model_name = '1x16_tanh_reg0p0_kin_directinput'
# model_name = '1x16_tanh_reg0p0_nomodel_ayonly_directinput_test_mlpsymmetric_wrong2'
# model_name = '1x16_tanh_reg0p0_nomodel_ayonly_directinput_test_mlpsymmetric'
# model_name = '1x16_tanh_reg0p0_nomodel_directinput_test_mlpsymmetric'
# model_name = '1x16_tanh_reg0p0_nomodel_directinput'
model_names = [
    # '1x16_tanh_reg0p0_nomodel_directinput',
    # '1x16_tanh_reg0p0_nomodel_directinput_test_mlpsymmetric',
    # '2x16_elu_reg0p0001_nomodel_directinput',
    # '2x16_elu_reg0p0001_nomodel_directinput_mlpsymmetric',
    # '2x24_elu_reg0p0001_nomodel_directinput',
    # '2x24_elu_reg0p0001_nomodel_directinput_mlpsymmetric',
    # '1x16_sigmoid_reg0p0001_nomodel_directinput',
    # '1x16_tanh_reg0p0_kin_directinput',
    '1x16_tanh_reg0p0_nomodel_directinput',
    '1x16_tanh_reg0p0_nomodel_directinput_fulldata_mlpsymmetric',
]

for num, model_name in enumerate(model_names):
    mlp = MultiLayerPerceptronMPC(model_name=model_name, predict_only=True)
    # mlp = MultiLayerPerceptronMPCSymmetric(model_name=model_name, predict_only=True)
    mlp.load_model()
    mlp.load_checkpoint('best')

    # plot this, ranges, constants  np.array([[5.,-0.1,0.5,0.25,-1.,2.]])
    ra_vx = ['vx',0,[0,10],5]
    ra_vy = ['vy',0,[-4,4],-0.0]
    ra_vtheta = ['vtheta',1,[-2,2],0.0]
    ra_beta = ['BETA',1,[-0.44,0.44],0.0]
    ra_ab = ['AB',0,[-8,2],-1.0]
    ra_tv = ['TV',0,[-1.5,1.5],0.0]
    disturbance_no = [0, 'disturbance ax']
    # disturbance_no = [1, 'disturbance ay']
    # disturbance_no = [2, 'disturbance atheta']
    params = [ra_vx, ra_vy, ra_vtheta, ra_beta, ra_ab, ra_tv]

    resolution = 100

    xy = []
    x = True
    input = np.array(())
    for (name, plot_this, range, constant) in params:
        if plot_this:
            data = np.linspace(range[0],range[1],resolution)
            if x:
                data = np.repeat(data, resolution)
                if len(input) > 0:
                    input = np.vstack((input, data))
                else:
                    input = data
                data = data.reshape((resolution, -1))
                xy.append([name, data])
                x = False
            else:
                data = np.tile(data, resolution)
                if len(input) > 0:
                    input = np.vstack((input, data))
                else:
                    input = data
                data = data.reshape((resolution, -1))
                xy.append([name, data])
        else:

            data = np.ones((1,resolution*resolution)) * constant

            if len(input) > 0:
                input = np.vstack((input,data))
            else:
                input = data
            # vxx = vxx.reshape((len(vx), -1))
            # betaa = betaa.reshape((len(beta), -1))
    input = input.T
    print(input)
    print(input.shape)
    input2 = np.array([[5.0,-0.1,0.5,0.25,-1.0,0.5]])
    # input2 = np.array([[5.,0.1,-0.5,-0.25,-1.,-2.]])
    print(mlp.predict(input2))
    disturbance = mlp.predict(input)
    print(disturbance)
    dist_plot = disturbance[:,disturbance_no[0]]
    dist_plot = dist_plot.reshape((resolution,-1))

    fig = plt.figure(num)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xy[0][1], xy[1][1], dist_plot, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xy[0][0])
    ax.set_ylabel(xy[1][0])
    ax.set_zlabel(disturbance_no[1])
    ax.set_title(model_name)

plt.show()



# random_state = 45
#
# # path_data_set = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/LearnedModel/20190625-200000_TF_filtered_vel/disturbance.pkl'
# # # path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190729-173735_trustworthy_mirrored_mpc'
# # dataframe = getPKL(path_data_set)
# # dataframe.pop('time [s]')
#
# # # Split data_set into training and test set
# # train_dataset = dataframe.sample(frac=0.8, random_state=random_state)
# # train_dataset = train_dataset.reset_index(drop=True)
# #
# # test_dataset = dataframe.drop(train_dataset.index)
# # test_dataset = test_dataset.reset_index(drop=True)
# #
# # train_labels = train_dataset[['disturbance vehicle ax local [m*s^-2]',
# #                                   'disturbance vehicle ay local [m*s^-2]',
# #                                   'disturbance pose atheta [rad*s^-2]']]
# # train_features = train_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
# #                                 'steer position cal [n.a.]', 'brake position effective [m]',
# #                                 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
# #
# # test_labels = test_dataset[['disturbance vehicle ax local [m*s^-2]',
# #                             'disturbance vehicle ay local [m*s^-2]',
# #                             'disturbance pose atheta [rad*s^-2]']]
# # test_features = test_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
# #                               'steer position cal [n.a.]', 'brake position effective [m]',
# #                               'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
#
#
# # model_name = '2x16_softplus_reg0p05_directinput_sym'
# # mlp = MultiLayerPerceptron(model_name=model_name, predict_only=True)
# # mlp.load_model()
# # mlp.load_checkpoint('best')
# # mlp.show_model_summary()
# # weights,biases = mlp.get_weights()
# # for i, (w1, b) in enumerate(zip(weights, biases)):
# #     print(w1.shape, '---', b.shape)
# #     # for line in w1:
# #     #     print(line)
#
# # test_errors = mlp.test_model(test_features, test_labels)
# # print(test_errors)
# # V:[0.5922 0.0379 0.0797], U:[0.1610 1.0844  0.02]
# # V=[9.20314860763906,1.5374833205406,-1.69869163682998]
# # U=[-0.262518217667255,-0.3935772773963,-1.97939283288554]
#
# # # inputs = [1,0.1,-0.5,-0.5,1,-0.1]
# # inputs = V+U
# # inputs_normed = mlp.normalize_data(inputs)
# # # x = inputs_normed.values
# # # for i, (w, b) in enumerate(zip(weights, biases)):
# # #     sol = np.matmul(w.transpose(), x) + b
# # #     if i < len(biases) - 1:
# # #         x = np.log(np.exp(sol) + 1)
# # #     else:
# # #         x = sol
# #
# # res = mlp.predict(np.array([inputs_normed.values]))
# # print(res)
# # mpc = DynamicVehicleMPC(direct_input=True)
# # v = list(inputs[:3])
# # u = list(inputs[3:])
# #
# # n = mpc.get_accelerations(v,u)
# # print(n)
# # result = np.add(np.array(x),np.array(n).transpose())
# # print(result)
# # ddriven = DataDrivenVehicleModel(model_name=model_name, direct_input=True)
# # res = ddriven.get_accelerations(v,u)
# # print(res)
# # q = np.linspace(-10,10)
# # y = np.log(np.exp(q) + 1)
# # plt.plot(q,y)
# # plt.show()
#
# # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190729-173735_trustworthy_mirrored_mpc')
# # train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
# # train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
# #                                  'turning angle [n.a]',
# #                                  'acceleration rear axle [m*s^-2]',
# #                                  'acceleration torque vectoring [rad*s^-2]']]
# # train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
# # train_labels = train_labels[['disturbance vehicle ax local [m*s^-2]',
# #                              'disturbance vehicle ay local [m*s^-2]',
# #                              'disturbance pose atheta [rad*s^-2]']]
# # test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
# # test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
# #                                  'turning angle [n.a]',
# #                                  'acceleration rear axle [m*s^-2]',
# #                                  'acceleration torque vectoring [rad*s^-2]']]
# #
# # test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))
# # test_labels = test_labels[['disturbance vehicle ax local [m*s^-2]',
# #                            'disturbance vehicle ay local [m*s^-2]',
# #                            'disturbance pose atheta [rad*s^-2]']]
#
# path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190806-111533_trustworthy_mirrored_mpc_newsplit (copy)/test_log_files')
# test_features = getPKL(os.path.join(path_data_set, '20190610T132237_00_sampledlogdata.pkl'))
# test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
#                                  'turning angle [n.a]',
#                                  'acceleration rear axle [m*s^-2]',
#                                  'acceleration torque vectoring [rad*s^-2]']]
#
# test_labels = getPKL(os.path.join(path_data_set, '20190610T132237_00_sampledlogdata.pkl'))
# test_labels = test_labels[['vehicle ax local [m*s^-2]',
#                            'vehicle ay local [m*s^-2]',
#                            'pose atheta [rad*s^-2]']]
# print(test_features.head())
# print(test_labels.head())
# # model_names = ['2x32_softplus_reg0p05_directinput', '5x64_relu_reg0p01_directinput']
# model_names = ['2x32_softplus_reg0p05_directinput_newsplit', '5x64_relu_reg0p01_directinput_newsplit']
#
# for model_name in model_names:
#     V = list(test_features.values[0][:3])
#     U = list(test_features.values[0][3:])
# #     def add_noise(x):
# #         mean = x
# #         std = abs(x) / 20.0
# #         return float(np.random.normal(mean, std, 1))
# #
# #     V = np.array([V for x in range(100)])
# #     U = np.array([U for x in range(100)])
# #     v_in = V.reshape(1, -1)
# #     res = list(map(add_noise, v_in[0]))
# #     V = np.array(res).reshape(-1, 3)
# #     plt.figure(1)
# #     plt.plot(V[:,0],'r')
# #     plt.plot(V[:,1],'b')
# #     plt.plot(V[:,2],'g')
# #     plt.plot(np.arctan(V[:,1]/V[:,0]),'m')
# #     plt.hold
# #
#     reference = test_labels.values[0]
#
#     vehicle_model = DataDrivenVehicleModel(model_name=model_name, direct_input=True)
#     # V = V.transpose()
#     # U = U.transpose()
#     print(V, U)
#     pred = vehicle_model.get_accelerations(V,U)
#     print(pred, reference)
#     error = reference - pred
#     mean_squared_error = np.mean(np.square(error), axis=0)
#     plt.figure(2)
#     plt.plot(error)
#     plt.hold
#     print(model_name, mean_squared_error)
# plt.show()
