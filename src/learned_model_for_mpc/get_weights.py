#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 25.06.19 18:19

@author: mvb
"""
import numpy as np

from learned_model_for_mpc.ml_models.mlp_keras_mpc import MultiLayerPerceptronMPC
from learned_model_for_mpc.ml_models.mlp_keras_sparse_mpc import MultiLayerPerceptronMPCSparse
from simulator.model.data_driven_model import DataDrivenVehicleModel
from simulator.model.no_model_sparse_model import NoModelSparseModel
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow_model_optimization.python.core.sparsity import keras as sparsity

# model_name = '2x16_softplus_reg0p1_directinput_sym' #symmetry no good for mpc
# model_name = '3x16_softplus_reg0p0001_kin_directinput_sym'  #symmetry no good for mpc
# model_name = '2x16_softplus_reg0p1_directinput' # on mpc as NN_11
# model_name = '3x16_softplus_reg0p0001_kin_directinput' # on mpc as NN_kin
# model_name = '2x16_softplus_reg0p01_directinput' # on mpc as NN_lessreg
# model_name = '1x16_tanh_reg0p01_directinput' # on mpc as NN_small_lessreg
# model_name = '1x16_tanh_reg0p0001_directinput'
# model_name = '0x6_None_reg0p01_directinput' # on mpc as NN_linear
# model_name = '0x1296_None_reg0p01_10ksample_l1_directinput'
# model_name = '0x1296_None_reg0p01_10ksample_expotrigopoly_l1_directinput'
# model_name = '0x1296_None_reg0p001_10ksample_expopoly3trigo_l1_directinput'
# model_name = '0x1503_None_reg0p01_10ksample_powerexpotrigopoly2_l1_directinput'
# model_name = '0x1503_None_reg0p1_10ksample_powerexpotrigopoly2_l1sparse_directinput'
# model_name = '0x1503_None_reg0p0001_100ksample_powerexpotrigopoly2_l1sparse_directinput'
# model_name = '2x16_tanh_reg0p01_kin_directinput' # on mpc as kinematic_model_2x16_tanh_reg0p0001
# model_name = '1x16_softplus_reg0p0001_kin_directinput' # on mpc as kinematic_model_1x16_softplus_reg0p0001
# model_name = '1x16_tanh_reg0p0001_kin_directinput' # on mpc as kinematic_model_1x16_tanh_reg0p0001
# model_name = '1x16_tanh_reg0p0_kin_directinput' # on mpc as kinematic_model_1x16_tanh_reg0p0
# model_name = '1x16_softplus_reg0p0_kin_directinput' # on mpc as kinematic_model_1x16_softplus_reg0p0
# model_name = '0x3168_None_reg0p0001_50ksample_powerexpotrigocustompoly2_l1sparse_directinput'
# model_name = '0x966_None_reg0p0001_50ksample_expotrigocustompoly2_l1sparse_directinput'
# model_name = '0x426_None_reg0p0001_50ksample_expotrigopoly2_l1sparse_directinput'
# model_name = '0x369_None_reg0p0001_50ksample_expotrigopoly2_l1sparse_directinput'
# model_name = '0x2161_None_reg0p0001_50ksample_expotrigopoly3_l1sparse_directinput'
# model_name = '0x2161_None_reg0p0001_50ksample_expotrigopoly3_l1sparse_axonly_directinput'
# model_name = '0x2161_None_reg0p0001_50ksample_expotrigopoly3_l1sparse_ayonly_directinput'
# model_name = '0x2161_None_reg0p0001_50ksample_expotrigopoly3_l1sparse_athetaonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly3_l1supersparse_axonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly3_l1supersparse_ayonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly3_l1supersparse_athetaonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1sparse_axonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1sparse_ayonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1sparse_athetaonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1supersparse_axonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1supersparse_ayonly_directinput'
# model_name = '0x12_None_reg0p0001_50ksample_expotrigopoly2_l1supersparse_athetaonly_directinput'
model_name = 'poly3_order3'

# if not 'sparse' in model_name:
#     print('not sparse')
#     mlp = MultiLayerPerceptronMPC(model_name=model_name, predict_only=True)
#     mlp.load_model()
#     mlp.load_checkpoint('best')
# else:
#     mlp = MultiLayerPerceptronMPCSparse(model_name=model_name, predict_only=True)
#     mlp.load_model()
#     # mlp.load_checkpoint(checkpoint='latest')
#     # mlp.load_checkpoint(checkpoint=-4) #for poly2_l1sparse_axonly
#     mlp.load_checkpoint(checkpoint=-26)
#     mlp.show_model_summary()
#     # mlp.pruned_model = sparsity.prune.strip_pruning(mlp.pruned_model)
#
# mlp.show_model_summary()
# weights, biases = mlp.get_weights()
# means, stds = mlp.get_normalizing_parameters()

# for i, (w, b) in enumerate(zip(weights, biases)):
    # print(w.shape, '---', b.shape)
    # for j, (weights,bias) in enumerate(zip(w.T,b)):
        # sorted_weights = weights
        # sorted_weights = np.abs(sorted_weights)
        # sorted_weights.sort()
        # indizes = np.where(np.abs(weights) > sorted_weights[0])[0]
        #
        # # percentiles = [np.percentile(weights, 5), np.percentile(weights, 95)]
        # # indizes_low = np.where(weights < percentiles[0])[0]
        # # indizes_high = np.where(weights > percentiles[1])[0]
        # # indizes = np.hstack((indizes_high, indizes_low))
        #
        # print('median:',np.median(weights))
        # print('std:',np.std(weights))
        # thresholds = [np.median(weights) - np.std(weights)/2.0, np.median(weights) + np.std(weights)/2.0]
        # indizes_low = np.where(weights < thresholds[0])[0]
        # indizes_high = np.where(weights > thresholds[1])[0]
        # indizes = np.hstack((indizes_high, indizes_low))
        #
        # print(len(indizes))
        # print_str = 'indizes ['
        # for index in indizes:
        #     print_str += f'{index}, '
        # print(print_str, ']')
        # # plt.figure(j)
        # # sns.distplot(weights[indizes], bins=50)
        # # print(mlp.train_stats.index.values[indizes])
        # res_list = np.array([(weights[indizes[0]], mlp.train_stats.index.values[indizes[0]])], dtype=[('score', np.float), ('name', 'U100')])
        # equation = 'x = '
        # equation += f'{weights[indizes[0]]} * ({mlp.train_stats.index.values[indizes[0]]} - {means[indizes[0]]}) / {stds[indizes[0]]} + '
        # for index in indizes[1:]:
        #     equation += f'{weights[index]} * ({mlp.train_stats.index.values[index]} - {means[index]}) / {stds[index]} + '
        #     res_list = np.vstack((res_list, np.array([(weights[index], mlp.train_stats.index.values[index])], dtype=[('score', np.float), ('name', 'U100')])))
        # equation = equation[:-3] + f' + {bias}'
        # print(equation)
        # res_list.sort(order='score', axis=0)
        # print(res_list)

#     w_lines = f'w{i+1} = ['
#     for line in w:
#         for value in line:
#             w_lines += str(value) + ' '
#         w_lines = w_lines[:-1]
#         w_lines += ';'
#     w_lines = w_lines[:-1] + '];'
#     print(w_lines)
#
#     b_lines = f'b{i+1} = ['
#     for line in b:
#         b_lines += str(line) + ' '
#     # b_lines = b_lines[:-1]
#     b_lines = b_lines[:-1] + '];'
#     print(b_lines)
# means_matlab = 'means = ['
# for value in means:
#     means_matlab += str(value) + ' '
# means_matlab = means_matlab[:-1]
# means_matlab = means_matlab[:-1] + '];'
# print(means_matlab)
# stds_matlab = 'stds = ['
# for value in stds:
#     stds_matlab += str(value) + ' '
# stds_matlab = stds_matlab[:-1]
# stds_matlab = stds_matlab[:-1] + '];'
# print(stds_matlab)
#
# inputs = np.array([[1,0.1,-0.5,-0.5,1,-0.1]])
# inputs = np.array([[5,0.1,-0.5,-0.25,-1,-3]])
# res = mlp.predict(inputs)

# # model = DataDrivenVehicleModel(model_type='hybrid_mlp', model_name=model_name, direct_input=True)
# model = DataDrivenVehicleModel(model_type='hybrid_kinematic_mlp', model_name=model_name, direct_input=True)
model = NoModelSparseModel(model_name=model_name)
# inputs = [5.,-0.1,0.5,0.25,-1.,3.] #[-1.53183965  4.04873158  8.52437451]
# res = model.get_accelerations(inputs[:3],inputs[3:])
# print(res)
# inputs = [5.,0.1,-0.5,-0.25,-1.,-3.] #[-1.53183965 -4.04873158 -8.52437451]
# res = model.get_accelerations(inputs[:3],inputs[3:])
# print(res)

inputs = np.array([[ 4.4730394 , -0.2678969 , -0.93233321, -0.25267067 ,-0.40952891, -1.16650583],
 [ 4.43005468, -0.26024857, -0.96622332, -0.25276112, -0.36382606, -1.21248537],
 [ 4.39318522, -0.24805991, -0.98425541, -0.25343907, -0.31258239, -1.26399591],
 [ 4.37660549, -0.23332697, -0.98210431, -0.25343907, -0.25689436, -1.31981247],
 [ 4.36999584, -0.20468512, -0.97481372, -0.25343907, -0.21999425, -1.35676526],
 [ 4.36496053, -0.16819296, -0.96806456, -0.25237094, -0.13397405, -1.4125736 ],
 [ 4.36914862, -0.14013288, -0.95019107, -0.24921754,  0.04142235, -1.25358771],
 [ 4.38302369, -0.1225921  ,-0.90805834, -0.23982114,  0.08121309, -1.21261827],
 [ 4.40772488, -0.11384263, -0.84579093, -0.21320129,  0.09350473, -1.19824096],
 [ 4.45223436, -0.11723932, -0.77100093, -0.20552054,  0.0915108 , -1.1967296 ]])

res = model.get_accelerations(inputs[:,:3],inputs[:,3:])
print(res)

inputs = [ 4.43005468, -0.26024857, -0.96622332, -0.25276112, -0.36382606, -1.21248537] #[-1.53183965 -4.04873158 -8.52437451]
res = model.get_accelerations(inputs[:3],inputs[3:])
print(res)