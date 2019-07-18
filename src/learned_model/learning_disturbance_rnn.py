#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.07.19 09:32

@author: mvb
"""
from data_visualization.data_io import getPKL
from learned_model.ml_models.rnn_keras import LongShortTermMemoryNetwork
from multiprocessing import Pool
import pandas as pd
import os
from data_visualization.data_io import getDirectories
import matplotlib.pyplot as plt


def main():
    # network_settings = []
    # i = 1
    # for layers in range(1, 10):
    #     for nodes_exp in range(2, 11):
    #         nodes_per_layer = 2 ** nodes_exp
    #         tot_params = layers * nodes_per_layer + (
    #                 layers - 1) * nodes_per_layer ** 2 + 7 * nodes_per_layer + 3 * nodes_per_layer + 3
    #         if tot_params < 35000:
    #             i += 1
    #             network_settings.append([layers, nodes_per_layer, 'relu', 0.0])
    #
    # for layers in range(10, 101, 5):
    #     for nodes_exp in range(2, 11):
    #         nodes_per_layer = 2 ** nodes_exp
    #         tot_params = layers * nodes_per_layer + (
    #                 layers - 1) * nodes_per_layer ** 2 + 7 * nodes_per_layer + 3 * nodes_per_layer + 3
    #         if tot_params < 35000:
    #             i += 1
    #             network_settings.append([layers, nodes_per_layer, 'relu', 0.0])
    # print(i, network_settings)

    network_settings = [
        [2, 32, 'relu', 0.1],
        [2, 32, 'tanh', 0.0],
        [2, 32, 'tanh', 0.01],
        [2, 32, 'tanh', 0.1],
        [1, 32, 'tanh', 0.0],
        [1, 32, 'tanh', 0.01],
        [1, 32, 'tanh', 0.1],
    ]
    if len(network_settings) > 1:
        chunks = [network_settings[i::8] for i in range(8)]
        pool = Pool(processes=8)

        pool.map(train_RNN, chunks)
    else:
        train_RNN(network_settings)

    get_loss_pictures()


def train_RNN(network_settings):
    random_state = 45

    # path_data_set = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/SequentialDataSets/20190709-121738_more_filtered_withlowspeed_learning_data'
    path_data_set = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/SequentialDataSets/20190710-151912_more_filtered_withlowspeed_learning_data'

    train_features = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_features = train_features[:,:,1:] #get rid of time values
    train_labels = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    test_features = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_features = test_features[:,:,1:] #get rid of time values
    test_labels = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))
    time_step = round(train_features[0, 1, 0] - train_features[0, 0, 0], 5)

    i = 1
    for l, npl, af, reg in network_settings:
        name = '{}x{}_{}_reg{}'.format(l, npl, af, str(reg).replace('.', 'p'))
        print('----> {}/{} Start training of model with name {}'.format(i, len(network_settings), name))
        lstm = LongShortTermMemoryNetwork(epochs=1000,
                                          learning_rate=1e-3,
                                          decay=1e-6,
                                          batch_size=128,
                                          input_sequence_length=train_features.shape[1],
                                          time_step=time_step,
                                          random_seed=random_state,
                                          model_name=name)

        lstm.build_new_model(layers=l, input_shape=train_features.shape[1:], nodes_per_layer=npl,
                             activation_function=af, regularization=reg)

        print(lstm.show_model_summary())

        lstm.train_model(train_features, train_labels)
        lstm.save_training_history()
        lstm.save_model_performance(test_features, test_labels)
        i += 1


def get_loss_pictures():
    root_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_rnn_models'

    folder_names = getDirectories(root_folder)
    folder_names = [name for name in folder_names if '2019' in name]
    # folder_names = ['/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_models/20190624-134333_3x32_relu_reg0p005',
    #                 '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_models/20190624-134333_3x32_relu_reg0p01']
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
    # network_settings = [
    #     [2, 32, 'relu', 0.0],
    #     # [5, 32, 'relu', 0.0],
    # ]
    main()
    # get_loss_pictures()
    # train_RNN(network_settings)
