#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 08.07.19 21:21

@author: mvb
"""
from dataanalysisV2.data_io import getDirectories, getPKL, dataframe_to_pkl, create_folder_with_time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
import numpy as np
import os
from collections import deque
import random
import pandas as pd

def get_sequential_disturbance(load_path_data=None, save_path_sequential_data=None, sequence_length=5, load_tag='test', test_portion=0.2, random_seed=42, tag='test'):
    if load_path_data == None:
        path_preprocessed_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'
        folders_preprocessed_data = getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(load_tag):
                defaultSim = str
                break
        load_path_data = path_preprocessed_data + '/' + defaultSim
    # file_path_data = load_path_data + '/' + 'dataset.pkl'

    # Dynamic MPC model
    vehicle_model = DynamicVehicleMPC()

    # # Dynamic MPC model modified
    # vehicle_model_name = '5x64_relu_reg0p0'
    # vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)

    # # Kinematic MPC model
    # vehicle_model = KinematicVehicleMPC()

    pklfiles = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                pklfiles.append([os.path.join(r, file), file])

    random.seed(random_seed)
    random.shuffle(pklfiles)

    train_files = pklfiles[:int(len(pklfiles)*(1-test_portion))]
    test_files = pklfiles[int(len(pklfiles)*(1-test_portion)):]

    train_features, train_labels = get_sequences(train_files, vehicle_model, sequence_length)
    test_features, test_labels = get_sequences(test_files, vehicle_model, sequence_length)

    save_folder_path = create_folder_with_time(save_path_sequential_data, tag=tag)

    file_path = save_folder_path + '/train_features.pkl'
    dataframe_to_pkl(file_path, train_features)
    file_path = save_folder_path + '/train_labels.pkl'
    dataframe_to_pkl(file_path, train_labels)
    file_path = save_folder_path + '/test_features.pkl'
    dataframe_to_pkl(file_path, test_features)
    file_path = save_folder_path + '/test_labels.pkl'
    dataframe_to_pkl(file_path, test_labels)

    print('Data set with disturbance saved to', save_folder_path)

def get_sequences(file_list, vehicle_model, sequence_length):
    sequential_data = []
    for file_path_data, file_name in file_list:
        # print('Loading file', file_path_data)
        dataframe = getPKL(file_path_data)
        dt = dataframe.values[1,0] - dataframe.values[0,0]
        data_set = dataframe.values[:,1:]
        velocities = data_set[:,:3]
        inputs = data_set[:,3:-3]
        target_output = data_set[:,-3:]

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

        dataframe = dataframe.drop('vehicle ax local [m*s^-2]', axis=1)
        dataframe = dataframe.drop('vehicle ay local [m*s^-2]', axis=1)
        dataframe = dataframe.drop('pose atheta [rad*s^-2]', axis=1)

        dataframe['disturbance vehicle ax local [m*s^-2]'] = output_disturbance[:,0]
        dataframe['disturbance vehicle ay local [m*s^-2]'] = output_disturbance[:,1]
        dataframe['disturbance pose atheta [rad*s^-2]'] = output_disturbance[:,2]

        sequence = deque(maxlen=sequence_length)

        for row in dataframe.values:
            sequence.append(row[:-3])
            if len(sequence) == sequence_length:
                sequential_data.append([sequence,row[-3:]])

    random.shuffle(sequential_data)

    features = []
    labels = []

    for sequence, disturbance in sequential_data:
        features.append(sequence)
        labels.append(disturbance)

    return np.array(features), np.array(labels)
    return 0,0


if __name__ == '__main__':
    get_sequential_disturbance(sequence_length=5, load_tag='lookatdata_more_filtered_trustnoreverse_1')