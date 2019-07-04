#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11.06.19 15:05

@author: mvb
"""
from dataanalysisV2.data_io import getDirectories, getPKL, dataframe_to_pkl
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
import numpy as np

def calculate_disturbance(load_path_data=None, tag='test'):
    if load_path_data == None:
        path_preprocessed_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/LearnedModel'
        folders_preprocessed_data = getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(tag):
                defaultSim = str
                break
        load_path_data = path_preprocessed_data + '/' + defaultSim
    file_path_data = load_path_data + '/' + 'dataset.pkl'

    print('Loading file', file_path_data)
    dataframe = getPKL(file_path_data)
    dt = dataframe.values[1,0] - dataframe.values[0,0]
    data_set = dataframe.values[:,1:]
    velocities = data_set[:,:3]
    inputs = data_set[:,3:-3]
    target_output = data_set[:,-3:]

    # # Dynamic MPC model
    # dynamic_mpc_model = DynamicVehicleMPC()
    # nominal_model_output = dynamic_mpc_model.get_accelerations(velocities, inputs)
    # nominal_model_output = np.vstack((nominal_model_output[0],nominal_model_output[1],nominal_model_output[2])).transpose()

    # Dynamic MPC model modified
    vehicle_model_name = '5x64_relu_reg0p0'
    vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)
    nominal_model_output = vehicle_model.get_accelerations(velocities, inputs)

    # # Kinematic MPC model
    # kinematic_mpc_model = KinematicVehicleMPC()
    # dBETA = np.append((inputs[1:,0]-inputs[:-1,0])/dt, (inputs[-1,0]-inputs[-2,0])/dt)
    # inputs = np.concatenate([inputs, dBETA[:, None]], axis=1)
    # nominal_model_output = kinematic_mpc_model.get_accelerations(velocities, inputs)

    output_disturbance = target_output - nominal_model_output

    dataframe = dataframe.drop('vehicle ax local [m*s^-2]', axis=1)
    dataframe = dataframe.drop('vehicle ay local [m*s^-2]', axis=1)
    dataframe = dataframe.drop('pose atheta [rad*s^-2]', axis=1)


    dataframe['disturbance vehicle ax local [m*s^-2]'] = output_disturbance[:,0]
    dataframe['disturbance vehicle ay local [m*s^-2]'] = output_disturbance[:,1]
    dataframe['disturbance pose atheta [rad*s^-2]'] = output_disturbance[:,2]

    file_path = load_path_data + '/disturbance.pkl'

    dataframe_to_pkl(file_path, dataframe)

    print('Data set with disturbance saved to', file_path)

if __name__ == '__main__':
    calculate_disturbance(tag='filtered_vel_data_driven_model')