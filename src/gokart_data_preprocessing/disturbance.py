#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11.06.19 15:05

@author: mvb
"""
import numpy as np
import os
import random
from collections import deque
import pandas as pd
import time

from config import directories
from data_visualization.data_io import getDirectories, getPKL, data_to_pkl, create_folder_with_time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel


def calculate_disturbance(load_path_data=None, data_set_name='test', test_portion=0.2, random_seed=42,
                          sequential=False, sequence_length=5):
    save_path_sequential_data = os.path.join(directories['root'], 'Data', 'MLPDatasets')
    if load_path_data == None:
        path_preprocessed_data = os.path.join(directories['root'], 'Data', 'Sampled')
        folders_preprocessed_data = getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(data_set_name):
                default_sim = str
                break
        load_path_data = path_preprocessed_data + '/' + default_sim

    # Dynamic MPC model
    vehicle_model = DynamicVehicleMPC()

    # # Dynamic MPC model modified
    # vehicle_model_name = '5x64_relu_reg0p0'
    # vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)

    # # Kinematic MPC model
    # vehicle_model = KinematicVehicleMPC()

    pkl_files = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                pkl_files.append([os.path.join(r, file), file])

    random.seed(random_seed)
    random.shuffle(pkl_files)

    train_files = pkl_files[:int(len(pkl_files) * (1 - test_portion))]
    test_files = pkl_files[int(len(pkl_files) * (1 - test_portion)):]

    train_features, train_labels = get_disturbance(train_files, vehicle_model, sequential, sequence_length)
    test_features, test_labels = get_disturbance(test_files, vehicle_model, sequential, sequence_length)

    save_folder_path = create_folder_with_time(save_path_sequential_data, tag=data_set_name)

    os.mkdir(os.path.join(save_folder_path, 'train_log_files'))
    os.mkdir(os.path.join(save_folder_path, 'test_log_files'))

    for path, name in train_files:
        os.popen('cp ' + path + ' ' + save_folder_path + '/train_log_files/' + name)
    for path, name in test_files:
        os.popen('cp ' + path + ' ' + save_folder_path + '/test_log_files/' + name)

    file_path = save_folder_path + '/train_features.pkl'
    data_to_pkl(file_path, train_features)
    file_path = save_folder_path + '/train_labels.pkl'
    data_to_pkl(file_path, train_labels)
    file_path = save_folder_path + '/test_features.pkl'
    data_to_pkl(file_path, test_features)
    file_path = save_folder_path + '/test_labels.pkl'
    data_to_pkl(file_path, test_labels)

    print('Data set with disturbance saved to', save_folder_path)


def get_disturbance(file_list, vehicle_model, sequential, sequence_length):
    features = []
    labels = []
    sequential_data = []
    starttime = time.time()
    for index, (file_path_data, file_name) in enumerate(file_list):
        print(int(index / len(file_list) * 100),
              '% completed.   current file:', file_name[:-4] + '   elapsed time:',
              int(time.time() - starttime), 's', end='\r')
        # print('Loading file', file_path_data)
        dataframe = getPKL(file_path_data)
        dt = dataframe.values[1, 0] - dataframe.values[0, 0]
        data_set = dataframe.values[:, 1:]
        velocities = data_set[:, :3]
        inputs = data_set[:, 3:-3]
        target_output = data_set[:, -3:]

        # Dynamic MPC model
        nominal_model_output = vehicle_model.get_accelerations(velocities, inputs)
        nominal_model_output = np.vstack(
            (nominal_model_output[0], nominal_model_output[1], nominal_model_output[2])).transpose()

        # # Dynamic MPC model modified
        # nominal_model_output = vehicle_model.get_accelerations(velocities, inputs)

        # # Kinematic MPC model
        # dBETA = np.append((inputs[1:,0]-inputs[:-1,0])/dt, (inputs[-1,0]-inputs[-2,0])/dt)
        # inputs = np.concatenate([inputs, dBETA[:, None]], axis=1)
        # nominal_model_output = vehicle_model.get_accelerations(velocities, inputs)

        output_disturbance = target_output - nominal_model_output

        # dataframe = dataframe.drop('vehicle ax local [m*s^-2]', axis=1)
        # dataframe = dataframe.drop('vehicle ay local [m*s^-2]', axis=1)
        # dataframe = dataframe.drop('pose atheta [rad*s^-2]', axis=1)

        dataframe = dataframe[['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                               'steer position cal [n.a.]', 'brake position effective [m]',
                               'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]

        dataframe['disturbance vehicle ax local [m*s^-2]'] = output_disturbance[:, 0]
        dataframe['disturbance vehicle ay local [m*s^-2]'] = output_disturbance[:, 1]
        dataframe['disturbance pose atheta [rad*s^-2]'] = output_disturbance[:, 2]
        if sequential:
            sequence = deque(maxlen=sequence_length)

            for row in dataframe.values:
                sequence.append(row[:-3])
                if len(sequence) == sequence_length:
                    sequential_data.append([np.array(sequence), row[-3:]])
        else:
            if len(features) == 0:
                features = dataframe.values[:, :-3]
                labels = dataframe.values[:, np.array([0,-3,-2,-1])]
            features = np.vstack((features, dataframe.values[:, :-3]))
            labels = np.vstack((labels, dataframe.values[:, np.array([0,-3,-2,-1])]))

    if sequential:
        random.shuffle(sequential_data)

        for sequence, disturbance in sequential_data:
            features.append(sequence)
            labels.append(disturbance)
        features = np.array(features)
        labels = np.array(labels)
    else:
        features = pd.DataFrame(np.array(features), columns=dataframe.columns[:-3])
        labels = pd.DataFrame(np.array(labels), columns=dataframe.columns[np.array([0,-3,-2,-1])])
    return features, labels


if __name__ == '__main__':
    calculate_disturbance(data_set_name='test', sequential=True)
