#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.06.19 11:09

@author: mvb
"""
from dataanalysisV2.data_io import getPKL
from learned_model.ml_model.mlp_keras import MultiLayerPerceptron
from learned_model.preprocessing.shuffle import shuffle_dataframe
import numpy as np


def main():
    epochs = 20

    path_data_set = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/LearnedModel/20190614-110109_TF_test/disturbance.pkl'
    dataframe = getPKL(path_data_set)
    dataframe.pop('time [s]')

    # Split data_set into training and test set
    train_dataset = dataframe.sample(frac=0.8, random_state=45)
    train_dataset = train_dataset.reset_index(drop=True)

    test_dataset = dataframe.drop(train_dataset.index)
    test_dataset = test_dataset.reset_index(drop=True)

    train_labels = train_dataset[['disturbance vehicle ax local [m*s^-2]',
                                  'disturbance vehicle ay local [m*s^-2]',
                                  'disturbance pose atheta [rad*s^-2]']]
    train_features = train_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                    'steer position cal [n.a.]', 'brake position effective [m]',
                                    'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]

    test_labels = test_dataset[['disturbance vehicle ax local [m*s^-2]',
                                'disturbance vehicle ay local [m*s^-2]',
                                'disturbance pose atheta [rad*s^-2]']]
    test_features = test_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                  'steer position cal [n.a.]', 'brake position effective [m]',
                                  'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]

    # train_stats = train_dataset.describe()
    # train_stats.pop('disturbance vehicle ax local [m*s^-2]')
    # train_stats.pop('disturbance vehicle ay local [m*s^-2]')
    # train_stats.pop('disturbance pose atheta [rad*s^-2]')
    # train_stats = train_stats.transpose()
    # print(train_stats)

    mlp = MultiLayerPerceptron(epochs=1000, learning_rate=1e-3, batch_size=100, random_seed=22, model_name='FirstTry')

    # mlp.load_model()
    mlp.build_new_model()

    print(mlp.show_model_summary())

    mlp.train_model(train_features, train_labels)
    mlp.save_training_history()

if __name__ == '__main__':
    main()