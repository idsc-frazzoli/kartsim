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


def calculate_disturbance(load_path_data=None, data_set_name='test', test_set_days=[], random_seed=42,
                          sequential=False, sequence_length=5, mirror_data=False, mpc_inputs=False):
    if sequential:
        save_path_data_set = os.path.join(directories['root'], 'Data', 'RNNDatasets')
    else:
        save_path_data_set = os.path.join(directories['root'], 'Data', 'MLPDatasets')

    default_data_set = None
    if load_path_data == None:
        path_preprocessed_data = os.path.join(directories['root'], 'Data', 'Sampled')
        folders_preprocessed_data = getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(data_set_name):
                default_data_set = str
                break
        if default_data_set is not None:
            load_path_data = path_preprocessed_data + '/' + default_data_set
        else:
            raise FileNotFoundError(
                f'No data with name {data_set_name} was found in path_preprocessed_data!\n Please specify.')

    if mpc_inputs:
        vehicle_model = DynamicVehicleMPC(direct_input=True)
    else:
        # Dynamic MPC model
        vehicle_model = DynamicVehicleMPC()

    # # Dynamic MPC model modified
    # vehicle_model_name = '5x64_relu_reg0p0'
    # vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)

    # # Kinematic MPC model
    # vehicle_model = KinematicVehicleMPC()

    pkl_files = []
    train_files = []
    test_files = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                # pkl_files.append([os.path.join(r, file), file])
                if file[:8] in test_set_days:
                    test_files.append([os.path.join(r, file), file])
                else:
                    train_files.append([os.path.join(r, file), file])

    # random.seed(random_seed)
    # random.shuffle(train_files)
    # random.shuffle(test_files)

    # Save the files in seperate folders
    save_folder_path = create_folder_with_time(save_path_data_set, tag=data_set_name)

    save_path_train_log_files = os.path.join(save_folder_path, 'train_log_files')
    save_path_test_log_files = os.path.join(save_folder_path, 'test_log_files')
    os.mkdir(save_path_train_log_files)
    os.mkdir(save_path_test_log_files)

    for path, name in train_files:
        os.popen('cp ' + path + ' ' + os.path.join(save_path_train_log_files, name))
    for path, name in test_files:
        os.popen('cp ' + path + ' ' + os.path.join(save_path_test_log_files, name))
    time.sleep(1)
    if mirror_data:
        # Generate mirrored log files (w.r.t. x-axis)
        mirror_logfiles(save_path_train_log_files)
        mirror_logfiles(save_path_test_log_files)

    if mpc_inputs:
        get_mpc_inputs(save_path_train_log_files)
        get_mpc_inputs(save_path_test_log_files)

    train_features, train_labels = get_disturbance(save_path_train_log_files, vehicle_model, sequential,
                                                   sequence_length, mpc_inputs)
    test_features, test_labels = get_disturbance(save_path_test_log_files, vehicle_model, sequential, sequence_length,
                                                 mpc_inputs)

    file_path = save_folder_path + '/train_features.pkl'
    data_to_pkl(file_path, train_features)
    file_path = save_folder_path + '/train_labels.pkl'
    data_to_pkl(file_path, train_labels)
    file_path = save_folder_path + '/test_features.pkl'
    data_to_pkl(file_path, test_features)
    file_path = save_folder_path + '/test_labels.pkl'
    data_to_pkl(file_path, test_labels)

    print('Data set with disturbance saved to', save_folder_path)


def get_disturbance(load_path_data, vehicle_model, sequential, sequence_length, mpc_inputs):
    file_list = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                file_list.append([os.path.join(r, file), file])
    file_list.sort()

    features = []
    labels = []
    sequential_data = []
    starttime = time.time()
    dataframe = pd.DataFrame()
    for index, (file_path_data, file_name) in enumerate(file_list):
        print(int(index / len(file_list) * 100),
              '% completed.   current file:', file_name[:-4] + '   elapsed time:',
              int(time.time() - starttime), 's', end='\r')
        # print('Loading file', file_path_data)
        dataframe = getPKL(file_path_data)
        dt = dataframe.values[1, 0] - dataframe.values[0, 0]
        # data_set = test_features.values[:, 4:]
        velocities = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]']].values
        if mpc_inputs:
            inputs = dataframe[
                ['turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                 'acceleration torque vectoring [rad*s^-2]']].values
        else:
            inputs = dataframe[
                ['steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]',
                 'motor torque cmd right [A_rms]']].values
        target_output = dataframe[
            ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].values
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

        # test_features = test_features.drop('vehicle ax local [m*s^-2]', axis=1)
        # test_features = test_features.drop('vehicle ay local [m*s^-2]', axis=1)
        # test_features = test_features.drop('pose atheta [rad*s^-2]', axis=1)
        if mpc_inputs:
            dataframe = dataframe[['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                   'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                                   'acceleration torque vectoring [rad*s^-2]']]
        else:
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
                labels = dataframe.values[:, np.array([0, -3, -2, -1])]
            features = np.vstack((features, dataframe.values[:, :-3]))
            labels = np.vstack((labels, dataframe.values[:, np.array([0, -3, -2, -1])]))

    if len(dataframe) != 0:
        if sequential:
            random.shuffle(sequential_data)

            for sequence, disturbance in sequential_data:
                features.append(sequence)
                labels.append(disturbance)
            features = np.array(features)
            labels = np.array(labels)
        else:
            features = pd.DataFrame(np.array(features), columns=dataframe.columns[:-3])
            labels = pd.DataFrame(np.array(labels), columns=dataframe.columns[np.array([0, -3, -2, -1])])
        return features, labels
    else:
        return dataframe, dataframe


def mirror_logfiles(load_path_data):
    pkl_files = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                pkl_files.append([os.path.join(r, file), file])
    pkl_files.sort()

    for file_path_data, file_name in pkl_files:
        dataframe = getPKL(file_path_data)
        dataframe['steer position cal [n.a.]'] = dataframe['steer position cal [n.a.]'] * -1
        dataframe['vehicle vy [m*s^-1]'] = dataframe['vehicle vy [m*s^-1]'] * -1
        dataframe['pose vtheta [rad*s^-1]'] = dataframe['pose vtheta [rad*s^-1]'] * -1
        dataframe['vehicle ay local [m*s^-2]'] = dataframe['vehicle ay local [m*s^-2]'] * -1
        dataframe['pose atheta [rad*s^-2]'] = dataframe['pose atheta [rad*s^-2]'] * -1
        dataframe['pose y [m]'] = dataframe['pose y [m]'] * -1
        dataframe['pose theta [rad]'] = dataframe['pose theta [rad]'] * -1
        temp_rimo_l = dataframe['motor torque cmd left [A_rms]'].copy()
        dataframe['motor torque cmd left [A_rms]'] = dataframe['motor torque cmd right [A_rms]']
        dataframe['motor torque cmd right [A_rms]'] = temp_rimo_l
        file_path = os.path.join(load_path_data, file_name[:-18] + 'mirrored_' + file_name[-18:])
        data_to_pkl(file_path, dataframe)


def get_mpc_inputs(load_path_data):
    pkl_files = []
    for r, d, f in os.walk(load_path_data):
        for file in f:
            if '.pkl' in file:
                pkl_files.append([os.path.join(r, file), file])
    pkl_files.sort()

    for file_path_data, file_name in pkl_files:
        dataframe = getPKL(file_path_data)
        dataframe = transform_inputs(dataframe)
        data_to_pkl(file_path_data, dataframe)


def transform_inputs(dataframe):
    turning_angle, acceleration_rear_axle, torque_tv = DynamicVehicleMPC().transform_inputs(
        dataframe['steer position cal [n.a.]'].values,
        dataframe['brake position effective [m]'].values,
        dataframe['motor torque cmd left [A_rms]'].values,
        dataframe['motor torque cmd right [A_rms]'].values,
        dataframe['vehicle vx [m*s^-1]'].values,
    )
    dataframe['turning angle [n.a]'] = turning_angle
    dataframe['acceleration rear axle [m*s^-2]'] = acceleration_rear_axle
    dataframe['acceleration torque vectoring [rad*s^-2]'] = torque_tv
    return dataframe


if __name__ == '__main__':
    calculate_disturbance(data_set_name='trustworthy_mirrored', sequential=False)
