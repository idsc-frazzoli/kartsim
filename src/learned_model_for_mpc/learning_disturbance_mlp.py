#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.06.19 11:09

@author: mvb
"""
import config
from data_visualization.data_io import getPKL
from learned_model_for_mpc.function_library import get_new_features
from learned_model_for_mpc.ml_models.mlp_keras_mpc import MultiLayerPerceptronMPC
# from learned_model_for_mpc.ml_models.mlp_keras_mpc_additfeatures import MultiLayerPerceptronMPCAdditFeatures
from learned_model_for_mpc.ml_models.mlp_keras_sparse_mpc import MultiLayerPerceptronMPCSparse
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
from data_visualization.data_io import getDirectories
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # network_settings = [
    #     [10, 64, 'tanh', 0.0],
    #     [20, 32, 'relu', 0.0],
    #     [50, 16, 'sigmoid', 0.0],
    # ]

    # network_settings = []
    # i=1
    # nodes_exp = 5
    # layers = 2
    # activ_func = 'softplus'
    # reg = 0.0001
    # for layers in range(1,10):
    #     for nodes_exp in range(6,11):
    #         for activ_func in ['elu', 'relu', 'tanh', 'sigmoid', 'softplus', 'exponential']:
    #         # for reg in [0.0001]:
    #             nodes = 2 ** nodes_exp
    #             tot_params = layers * nodes + (layers - 1) * nodes ** 2 + 7 * nodes + 3 * nodes + 3
    #             if tot_params < 35000:
    #                 i += 1
    #                 network_settings.append([layers,nodes,activ_func,reg])
    # print(i)
    # print(network_settings)

    network_settings = [
        # # [2, 32, 'elu', 0.0001],
        # # [2, 32, 'relu', 0.0001],
        # # [2, 32, 'tanh', 0.0001],
        # # [2, 32, 'sigmoid', 0.0001],
        # # [2, 16, 'softplus', 0.1],
        # [1, 16, 'tanh', 0.1],
        # [1, 16, 'tanh', 0.01],
        # [1, 16, 'tanh', 0.001],
        # [1, 16, 'tanh', 0.0001],
        # [1, 16, 'tanh', 0.0],
        # [1, 16, 'softplus', 0.1],
        # [1, 16, 'softplus', 0.01],
        # [1, 16, 'softplus', 0.001],
        # [1, 16, 'softplus', 0.0001],
        # [1, 16, 'softplus', 0.0],
        # [0, 1296, None, 0.0],
        # [0, 966, None, 0.1],
        # [0, 966, None, 0.01],
        # [0, 99, None, 0.0001],
        # [1, 32, None, 0.0],
        # [1, 16, 'softplus', 0.0],
        [1, 16, 'tanh', 0.0],
        # [1, 16, None, 0.0],
        # [2, 32, None, 0.0],
    ]
    train_features = train_labels = test_features = test_labels = None
    # train_features, train_labels, test_features, test_labels = get_data_set_for_sparsityNN()
    train_features, train_labels, test_features, test_labels = get_data_set_for_nomodel_morefeatures()

    if len(network_settings) >= 5:
        chunks = [network_settings[i::5] for i in range(5)]
        if train_features is not None:
            for i,chunk in enumerate(chunks):
                chunks[i] = chunk + [train_features, train_labels, test_features, test_labels]
        pool = Pool(processes=5)
        pool.map(train_NN, chunks)
    elif len(network_settings) > 1:
        no_settings = len(network_settings)
        chunks = [network_settings[i::no_settings] for i in range(no_settings)]
        if train_features is not None:
            for i,chunk in enumerate(chunks):
                chunks[i] = chunk + [train_features, train_labels, test_features, test_labels]
        pool = Pool(processes=no_settings)
        pool.map(train_NN, chunks)
    else:
        if train_features is not None:
            network_settings = network_settings + [train_features, train_labels, test_features, test_labels]
        train_NN(network_settings)



def train_NN(network_settings):
    random_state = 45

    if len(network_settings) == 5:
        test_labels = network_settings.pop()
        test_features = network_settings.pop()
        train_labels = network_settings.pop()
        train_features = network_settings.pop()
    else:
        # train_features, train_labels, test_features, test_labels = get_data_set()
        raise ValueError
    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    print(test_labels.shape)
    print(len(train_features.columns.values))
    for i, name in enumerate(train_features.columns.values):
        print(name)

    # i = 1
    # for l, npl, af, reg in network_settings:
    #     # name = '{}x{}_{}_reg{}_kin_directinput'.format(l, npl, str(af), str(reg).replace('.', 'p'))
    #     name = '{}x{}_{}_reg{}_nomodel_morefeatures_from_poly3reduced_directinput'.format(l, npl, af, str(reg).replace('.', 'p'))
    #     print('----> {}/{} Start training of model with model_type {}'.format(i, len(network_settings), name))
    #     mlp = MultiLayerPerceptronMPC(epochs=1000, learning_rate=1e-3, batch_size=100, random_seed=random_state,
    #                                   model_name=name)
    #     mlp.build_new_model(input_dim=train_features.shape[1], output_dim=train_labels.shape[1], layers=l, nodes_per_layer=npl, activation_function=af, regularization=reg)
    #     mlp.train_model(train_features, train_labels)
    #     mlp.save_training_history()
    #     mlp.save_model_performance(test_features, test_labels)
    #
    #     # # name = '{}x{}_{}_reg{}_50ksample_expotrigopoly3reduced_l1sparse_ayonly_directinput'.format(l, npl, af, str(reg).replace('.', 'p'))
    #     # print('----> {}/{} Start training of model with model_type {}'.format(i, len(network_settings), name))
    #     # # mlp = MultiLayerPerceptronMPCAdditFeatures(epochs=100000, learning_rate=1e-4, batch_size=100,
    #     # #                                            random_seed=random_state,
    #     # #                                            model_name=name)
    #     # mlp = MultiLayerPerceptronMPCSparse(epochs=500, learning_rate=1e-4, batch_size=100,
    #     #                                            random_seed=random_state,
    #     #                                            model_name=name)
    #     # mlp.build_train_test_model(train_features, train_labels, test_features=test_features, test_labels=test_labels,
    #     #                            layers=l, nodes_per_layer=npl, activation_function=af, regularization=reg)
    #     # #
    #     # print(f'{name} done with training! :)')
    #     # i += 1

def get_data_set_for_nomodel_morefeatures():
    # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190829-091021_final_data_set_dynamic_directinput')
    # path_data_set = os.path.join(config.directories['root'],
    #                              'Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput')
    path_data_set = os.path.join(config.directories['root'],
                                 'Data/MLPDatasets/20190829-092236_final_data_set_nomodel_directinput')
    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))

    merged_train = train_features.join(train_labels.iloc[:, 1:])
    merged_train = merged_train.sample(n=50000, random_state=16)
    train_features = merged_train.iloc[:, :-3]
    train_labels = merged_train.iloc[:, -3:]

    merged_test = test_features.join(test_labels.iloc[:,1:])
    merged_test = merged_test.sample(n=30000, random_state=42)
    test_features = merged_test.iloc[:, :-3]
    test_labels = merged_test.iloc[:, -3:]

    train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'turning angle [n.a]',
                                     'acceleration rear axle [m*s^-2]',
                                     'acceleration torque vectoring [rad*s^-2]']]
    train_features = get_new_features(train_features)
    # train_labels = train_labels[['disturbance vehicle ax local [m*s^-2]',
    #                              'disturbance vehicle ay local [m*s^-2]',
    #                              'disturbance pose atheta [rad*s^-2]']]
    test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                   'turning angle [n.a]',
                                   'acceleration rear axle [m*s^-2]',
                                   'acceleration torque vectoring [rad*s^-2]']]
    test_features = get_new_features(test_features)
    # test_labels = test_labels[['disturbance vehicle ax local [m*s^-2]',
    #                            'disturbance vehicle ay local [m*s^-2]',
    #                            'disturbance pose atheta [rad*s^-2]']]

    train_labels = train_labels[
        ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)
    test_labels = test_labels[
        ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)

    train_features = train_features.iloc[:, [1, 2, 4, 9, 15, 20, 23, 24, 25, 26, 27, 29, 33, 36, 37, 44, 45, 46, 47, 48, 56, 58, 59, 65, 69, 73, 79]]
    test_features = test_features.iloc[:, [1, 2, 4, 9, 15, 20, 23, 24, 25, 26, 27, 29, 33, 36, 37, 44, 45, 46, 47, 48, 56, 58, 59, 65, 69, 73, 79]]

    return train_features, train_labels, test_features, test_labels

def get_data_set_for_sparsityNN():
    # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190829-091021_final_data_set_dynamic_directinput')
    # path_data_set = os.path.join(config.directories['root'],
    #                              'Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput')
    path_data_set = os.path.join(config.directories['root'],
                                 'Data/MLPDatasets/20190829-092236_final_data_set_nomodel_directinput')
    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))

    merged_train = train_features.join(train_labels.iloc[:, 1:])
    merged_train = merged_train.sample(n=50000, random_state=16)
    train_features = merged_train.iloc[:, :-3]
    train_labels = merged_train.iloc[:, -3:]

    merged_test = test_features.join(test_labels.iloc[:,1:])
    merged_test = merged_test.sample(n=30000, random_state=42)
    test_features = merged_test.iloc[:, :-3]
    test_labels = merged_test.iloc[:, -3:]

    train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'turning angle [n.a]',
                                     'acceleration rear axle [m*s^-2]',
                                     'acceleration torque vectoring [rad*s^-2]']]
    train_features = get_new_features(train_features)
    # train_labels = train_labels[['disturbance vehicle ax local [m*s^-2]',
    #                              'disturbance vehicle ay local [m*s^-2]',
    #                              'disturbance pose atheta [rad*s^-2]']]
    test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                   'turning angle [n.a]',
                                   'acceleration rear axle [m*s^-2]',
                                   'acceleration torque vectoring [rad*s^-2]']]
    test_features = get_new_features(test_features)
    # test_labels = test_labels[['disturbance vehicle ax local [m*s^-2]',
    #                            'disturbance vehicle ay local [m*s^-2]',
    #                            'disturbance pose atheta [rad*s^-2]']]

    # train_labels = train_labels[
    #     ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)
    # test_labels = test_labels[
    #     ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)


    train_labels = train_labels[
        ['vehicle ax local [m*s^-2]']].reset_index(drop=True)
    test_labels = test_labels[
        ['vehicle ax local [m*s^-2]']].reset_index(drop=True)
    # # feature selection for poly2_l1supersparse
    # train_features = train_features.iloc[:, [4, 7, 16, 20, 28, 37, 39, 45, 46, 137, 161, 163, 190, 220, 239, 242, 6, 15, 62, 108, 109, 120, 154, 156, 200, 231, 234, 243, 249, 250, ]]
    # test_features = test_features.iloc[:, [4, 7, 16, 20, 28, 37, 39, 45, 46, 137, 161, 163, 190, 220, 239, 242, 6, 15, 62, 108, 109, 120, 154, 156, 200, 231, 234, 243, 249, 250, ]]
    # #feature selection for poly3_l1supersparse
    # train_features = train_features.iloc[:, [12, 20, 162, 549, 551, 553, 1268, 1961, 2023, 1669, 1824, 1856,  ]]
    # test_features = test_features.iloc[:, [12, 20, 162, 549, 551, 553, 1268, 1961, 2023, 1669, 1824, 1856,  ]]

    # train_labels = train_labels[
    #     ['vehicle ay local [m*s^-2]']].reset_index(drop=True)
    # test_labels = test_labels[
    #     ['vehicle ay local [m*s^-2]']].reset_index(drop=True)
    # # feature selection for poly2_l1supersparse
    # train_features = train_features.iloc[:,
    #                  [23, 51, 53, 61, 70, 77, 78, 99, 143, 181, 196, 203, 204, 226, 247, 1, 2, 24, 27, 49, 50, 57, 58, 73, 81, 100, 110, 151, 173, 209,]]
    # test_features = test_features.iloc[:,
    #                 [23, 51, 53, 61, 70, 77, 78, 99, 143, 181, 196, 203, 204, 226, 247, 1, 2, 24, 27, 49, 50, 57, 58, 73, 81, 100, 110, 151, 173, 209, ]]
    #feature selection for poly3_l1supersparse
    # train_features = train_features.iloc[:,[485, 537, 648, 909, 1061, 705, 707, 721, 741, 871, 1525, 1685,  ]]
    # test_features = test_features.iloc[:,[485, 537, 648, 909, 1061, 705, 707, 721, 741, 871, 1525, 1685,  ]]

    # train_labels = train_labels[
    #     ['pose atheta [rad*s^-2]']].reset_index(drop=True)
    # test_labels = test_labels[
    #     ['pose atheta [rad*s^-2]']].reset_index(drop=True)
    # feature selection for poly2_l1supersparse
    # train_features = train_features.iloc[:,
    #                  [11, 25, 33, 43, 49, 61, 86, 97, 100, 135, 179, 195, 196, 212, 215, 245, 262, 2, 18, 58, 75, 77, 78, 80, 128, 153, 171, 240, 247, 260,]]
    # test_features = test_features.iloc[:,
    #                 [11, 25, 33, 43, 49, 61, 86, 97, 100, 135, 179, 195, 196, 212, 215, 245, 262, 2, 18, 58, 75, 77, 78, 80, 128, 153, 171, 240, 247, 260,]]
    #feature selection for poly3_l1supersparse
    # train_features = train_features.iloc[:,[343, 457, 485, 741, 1059, 1367, 77, 79, 327, 889, 909, 918,  ]]
    # test_features = test_features.iloc[:,[343, 457, 485, 741, 1059, 1367, 77, 79, 327, 889, 909, 918,  ]]

    # for name in train_features.columns.values:
    #     print(name)
    # train_features = train_features.sort_values(by=['(pose vtheta [rad*s^-1])^(1/2)'])
    # print(train_features['(pose vtheta [rad*s^-1])^(1/2)'].head())
    # print(train_features['pose vtheta [rad*s^-1]'].head())
    # print(train_features['(pose vtheta [rad*s^-1])^(1/2)'].tail())
    # print(train_features['pose vtheta [rad*s^-1]'].tail())

    return train_features, train_labels, test_features, test_labels

def get_data_set_for_kinematic_model():
    # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190829-091021_final_data_set_dynamic_directinput')
    path_data_set = os.path.join(config.directories['root'],
                                 'Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput')
    # path_data_set = os.path.join(config.directories['root'],
    #                              'Data/MLPDatasets/20190829-092236_final_data_set_nomodel_directinput')
    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))

    train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'turning angle [n.a]',
                                     'acceleration rear axle [m*s^-2]',
                                     'acceleration torque vectoring [rad*s^-2]']]
    # train_features = get_new_features(train_features)
    train_labels = train_labels[['disturbance vehicle ax local [m*s^-2]',
                                 'disturbance vehicle ay local [m*s^-2]',
                                 'disturbance pose atheta [rad*s^-2]']]
    test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                   'turning angle [n.a]',
                                   'acceleration rear axle [m*s^-2]',
                                   'acceleration torque vectoring [rad*s^-2]']]
    # test_features = get_new_features(test_features)
    test_labels = test_labels[['disturbance vehicle ax local [m*s^-2]',
                               'disturbance vehicle ay local [m*s^-2]',
                               'disturbance pose atheta [rad*s^-2]']]

    return train_features, train_labels, test_features, test_labels

def get_data_set_for_nomodel_purestatesandinputs():
    # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190829-091021_final_data_set_dynamic_directinput')
    # path_data_set = os.path.join(config.directories['root'],
    #                              'Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput')
    path_data_set = os.path.join(config.directories['root'],
                                 'Data/MLPDatasets/20190829-092236_final_data_set_nomodel_directinput')
    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))

    train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'turning angle [n.a]',
                                     'acceleration rear axle [m*s^-2]',
                                     'acceleration torque vectoring [rad*s^-2]']]

    test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                   'turning angle [n.a]',
                                   'acceleration rear axle [m*s^-2]',
                                   'acceleration torque vectoring [rad*s^-2]']]

    train_labels = train_labels[
        ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)
    test_labels = test_labels[
        ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].reset_index(drop=True)


    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    main()
    # get_loss_pictures()
    # network_settings = [
    #     # [2, 32, 'softplus', 0.0],
    #     [5, 64, 'relu', 0.01],
    # ]
    # train_NN(network_settings)
