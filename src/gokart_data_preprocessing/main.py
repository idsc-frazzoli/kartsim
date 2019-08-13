#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
import os
import time

from config import directories, filenames
from data_visualization.data_io import create_folder_with_time, dict_keys_to_pkl
from gokart_data_preprocessing.tag_raw_data import TagRawData
from gokart_data_preprocessing.prepare_data import prepare_dataset
from gokart_data_preprocessing.sample_from_data import sample_from_logdata
from gokart_data_preprocessing.disturbance import calculate_disturbance


def main():
    t = time.time()

    # ____________User parameters____________
    # Choose a name for the data set
    dataset_name = 'trustworthy_mirrored_mpc'
    random_seed = 42

    # Choose data that should be contained in the data set
    # for a list of available signals can be found in the gokart_raw_data.py
    required_data_list = [
        'pose x [m]',
        'pose y [m]',
        'pose theta [rad]',
        'vehicle vx [m*s^-1]',
        'vehicle vy [m*s^-1]',
        'pose vtheta [rad*s^-1]',
        'vehicle ax local [m*s^-2]',
        'vehicle ay local [m*s^-2]',
        'pose atheta [rad*s^-2]',
        'steer position cal [n.a.]',
        'brake position effective [m]',
        'motor torque cmd left [A_rms]',
        'motor torque cmd right [A_rms]'
    ]

    # Choose the properties you wish the data to have (required_tags_list) or not to have (exclusion_tags_list)
    # possible tags to choose from are: ('multiple laps', 'high slip angles', 'reverse', 'trustworthy data', 'pose quality')
    required_tags_list = [
        'trustworthy data',
        # 'high slip angles',
        # 'multiple laps',
        # 'pose quality',
    ]  # list of signals and tags which should be true for the logs used to build the dataset
    exclusion_tags_list = [
        'reverse',
    ]

    # ______________________
    # check all the raw logs for missing/incomplete data and tag them for other characteristics
    redo_data_tagging = False # only needs to be done once

    # Filter data and compute inferred data from raw logs
    filter_data = True

    # Sample data
    sample_data = True
    # Sampling time period used for sampling the raw data
    sampling_time_period = 0.1  # [s]

    # Calculate disturbance (difference between the nominal model's acceleration estimation and the measured acceleration from log data)
    mlp_data_set = True #needs filter_data and sample_data to be True
    # Choose portion of test data
    mlp_test_set_days = ['20190701', '20190708', '20190709', '20190711', '20190719', '20190729']

    # Calculate sequential disturbance (difference between the nominal model's acceleration estimation and the measured acceleration from log data)
    lstm_data_set = False #needs filter_data and sample_data to be True
    # Specify the number of past states considered by the LSTM
    sequence_length = 5
    # Choose portion of test data
    lstm_test_portion = 0.2

    # ______________________
    # path where all the raw logfiles are
    root_path_raw_data = directories['rawgokartdata']

    path_sampled_data = None
    path_preprocessed_dataset = None
    # ______________^^^_________________
    # Load data tags
    data_tagging = TagRawData(root_path_raw_data)

    if not os.path.exists(os.path.join(root_path_raw_data, filenames['rawdatatags'])) and not redo_data_tagging:
        data_tagging = TagRawData(root_path_raw_data)
        data_tagging.tag_log_files(overwrite=True)
        data_tagging.save_prepro_params()

    if filter_data:
        data_tagging.sort_out_data(required_data_list, required_tags_list, exclusion_tags_list)
        print('Data preprocessing started...')
        filtered_data_dict = prepare_dataset(root_path_raw_data, data_tagging, required_data_list,
                                             start_from='20190514')
        print('Data preprocessing completed.')

        # parent directory where the preprocessed data should be saved to (separate folder will be created in this directory)
        save_path_filtered_data = os.path.join(directories['root'], 'Data', 'Filtered')
        path_preprocessed_dataset = create_folder_with_time(save_path_filtered_data, dataset_name)
        print('Now writing to file at', path_preprocessed_dataset)
        dict_keys_to_pkl(filtered_data_dict, path_preprocessed_dataset)

    if sample_data:
        print('Sampling Data...')
        path_sampled_data = sample_from_logdata(sampling_time_period, path_preprocessed_dataset,
                                                dataset_name, merge_data=False)
    if mlp_data_set:
        if path_sampled_data is None:
            path_sampled_data = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190717-211005_high_slip_angles/'

        print('Calcluating disturbance on predicitons with nominal vehicle model...')
        calculate_disturbance(path_sampled_data, data_set_name=dataset_name, test_portion=mlp_test_portion,
                              random_seed=random_seed, sequential=False, mirror_data=True, mpc_inputs=True)

    if lstm_data_set:
        print('Get sequential disturbance data...')
        calculate_disturbance(path_sampled_data, data_set_name=dataset_name, test_portion=lstm_test_portion, random_seed=random_seed,
                              sequential=True, sequence_length=sequence_length, mirror_data=True, mpc_inputs=False)

    print('Total computing time: ', int(time.time() - t), "s")


if __name__ == "__main__":
    main()
