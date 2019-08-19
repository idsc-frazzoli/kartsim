#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.06.19 11:09

@author: mvb
"""
import config
from data_visualization.data_io import getPKL
from learned_model.ml_models.mlp_keras import MultiLayerPerceptron
from multiprocessing import Pool
import pandas as pd
import os
from data_visualization.data_io import getDirectories
import matplotlib.pyplot as plt
import seaborn as sns
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC

def main():
    # network_settings = [
    #     [10, 64, 'tanh', 0.0],
    #     [20, 32, 'relu', 0.0],
    #     [50, 16, 'sigmoid', 0.0],
    # ]

    network_settings = []

    # i=1
    # for layers in range(1,10):
    #     for nodes_exp in range(2,11):
    #         nodes = 2 ** nodes_exp
    #         tot_params = layers * nodes + (layers - 1) * nodes ** 2 + 7 * nodes + 3 * nodes + 3
    #         if tot_params < 35000:
    #             i += 1
    #             network_settings.append([layers,nodes,'relu',0.0])
    #
    # for layers in range(10,101,5):
    #     for nodes_exp in range(2,11):
    #         nodes = 2 ** nodes_exp
    #         tot_params = layers * nodes + (layers - 1) * nodes ** 2 + 7 * nodes + 3 * nodes + 3
    #         if tot_params < 35000:
    #             i += 1
    #             network_settings.append([layers,nodes,'relu',0.0])
    # print(i,network_settings)

    network_settings = [
        # [1, 32, 'softplus', 0.001],
        # [1, 32, 'softplus', 0.01],
        # [1, 32, 'softplus', 0.05],
        # [1, 32, 'softplus', 0.1],
        # [2, 32, 'softplus', 0.001],
        # [2, 32, 'softplus', 0.01],
        [2, 32, 'softplus', 0.05],
        [2, 32, 'softplus', 0.0001],
        # [2, 32, 'softplus', 0.1],
        # [1, 32, 'tanh', 0.001],
        # [1, 32, 'tanh', 0.01],
        # [1, 32, 'tanh', 0.05],
        # [1, 32, 'tanh', 0.1],
        # [2, 32, 'tanh', 0.001],
        # [2, 32, 'tanh', 0.01],
        # [2, 32, 'tanh', 0.05],
        # [2, 32, 'tanh', 0.1],
    ]

    if len(network_settings) >= 5:
        chunks = [network_settings[i::5] for i in range(5)]
        pool = Pool(processes=5)
        pool.map(train_NN, chunks)
    elif len(network_settings) > 1:
        no_settings = len(network_settings)
        chunks = [network_settings[i::no_settings] for i in range(no_settings)]
        pool = Pool(processes=no_settings)
        pool.map(train_NN, chunks)
    else:
        train_NN(network_settings)

    # get_loss_pictures()


def train_NN(network_settings):
    random_state = 45

    path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190813-141626_trustworthy_bigdata')
    # path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190717-100934_trustworthy_data')
    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_features = train_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                    'steer position cal [n.a.]', 'brake position effective [m]',
                                    'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']] # get rid of time values
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    train_labels = train_labels[['disturbance vehicle ax local [m*s^-2]',
                                  'disturbance vehicle ay local [m*s^-2]',
                                  'disturbance pose atheta [rad*s^-2]']]
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_features = test_features[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'steer position cal [n.a.]', 'brake position effective [m]',
                                     'motor torque cmd left [A_rms]',
                                     'motor torque cmd right [A_rms]']]  # get rid of time values
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))
    test_labels = test_labels[['disturbance vehicle ax local [m*s^-2]',
                                 'disturbance vehicle ay local [m*s^-2]',
                                 'disturbance pose atheta [rad*s^-2]']]


    i = 1
    for l, npl, af, reg in network_settings:
        if 'mirrored' in path_data_set:
            name = '{}x{}_{}_reg{}_m'.format(l, npl, af, str(reg).replace('.', 'p'))
        else:
            name = '{}x{}_{}_reg{}_sym'.format(l, npl, af, str(reg).replace('.', 'p'))
        print('----> {}/{} Start training of model with model_type {}'.format(i, len(network_settings), name))
        mlp = MultiLayerPerceptron(epochs=1000, learning_rate=1e-3, batch_size=100, random_seed=random_state,
                                   model_name=name)

        # mlp.load_model()
        mlp.build_new_model(layers=l, nodes_per_layer=npl, activation_function=af, regularization=reg)
        # mlp.build_new_model(layers=2, nodes_per_layer=32, activation_function='relu', regularization=0.01)

        # print(mlp.show_model_summary())

        mlp.train_model(train_features, train_labels)
        mlp.save_training_history()
        mlp.save_model_performance(test_features, test_labels)
        i += 1


def get_loss_pictures():
    # root_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_models'
    root_folder = os.path.join(config.directories['root'], 'Models/trained_mlp_models')

    folder_names = getDirectories(root_folder)
    folder_names.remove('logs')
    for name in folder_names:
        train_history_dir = os.path.join(root_folder, name, 'training_history')

        if not any([file.endswith('.pdf') for file in os.listdir(train_history_dir)]):
            hist = pd.DataFrame.from_csv(os.path.join(train_history_dir, 'history.csv'))
            plt.figure('Mean Squared Error (Loss)')
            plt.plot(hist[['epoch']].values, hist[['loss', 'val_loss']].values)
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.savefig(os.path.join(train_history_dir, 'loss_mean_squ_err.pdf'))

            plt.figure('Mean Absolute Error')
            plt.plot(hist[['epoch']].values, hist[['mean_absolute_error', 'val_mean_absolute_error']].values)
            plt.legend(['Training Error', 'Validation Error'])
            plt.savefig(os.path.join(train_history_dir, 'mean_abs_err.pdf'))

            plt.figure('Coefficient of Determination R^2')
            plt.plot(hist[['epoch']].values, hist[['coeff_of_determination', 'val_coeff_of_determination']].values)
            plt.legend(['Training', 'Validation'])
            plt.savefig(os.path.join(train_history_dir, 'R_squared.pdf'))

            plt.close('all')

    # _path = os.path.join(model_dir, 'training_history')
    # hist_save_path = os.path.join(save_path, 'history.csv')
    # hist.to_csv(hist_save_path)


if __name__ == '__main__':
    main()
    # get_loss_pictures()
    # network_settings = [
    #     [5, 64, 'relu', 0.01],
    # ]
    # train_NN(network_settings)
