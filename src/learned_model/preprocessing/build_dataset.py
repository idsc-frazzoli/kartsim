#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 31.05.19 08:45

@author: mvb
"""
from dataanalysisV2.data_io import create_folder_with_time, dict_keys_to_pkl
from dataanalysisV2.gokartpreprocessing.sample_from_data import sample_from_logdata
from dataanalysisV2.gokartpreprocessing.prepare_data import prepare_dataset
from dataanalysisV2.gokartpreprocessing.tag_raw_data import TagRawData
from dataanalysisV2.gokartpreprocessing.merge_data import merge_data
from learned_model.preprocessing.calculate_disturbance import calculate_disturbance

import time


def main():
    t = time.time()

    # ________User parameters

    pathRootData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts'  # path where all the raw logfiles are

    #______________________
    # check and tag all the raw logs for missing/incomplete data and other characteristics
    # tag_data = True
    tag_data = False

    #______________________
    # Filter data and compute inferred data from raw logs
    filter_data = True
    # filter_data = False
    # list of signals which should be true for the logs used to build the dataset
    required_data_list = ['vehicle vx [m*s^-1]',
                          'vehicle vy [m*s^-1]',
                          'pose vtheta [rad*s^-1]',
                          'steer position cal [n.a.]',
                          'brake position effective [m]',
                          'motor torque cmd left [A_rms]',
                          'motor torque cmd right [A_rms]',
                          'vehicle ax local [m*s^-2]',
                          'vehicle ay local [m*s^-2]',
                          'pose atheta [rad*s^-2]'
                          ]
    # list of tags which should be true for the logs used to build the dataset
    required_tags_list = [
        # 'multiple laps',
        # 'high slip angles',
        'trustworthy data'
    ]
    # list of signals or tags which should not be true for the logs used to build the dataset
    exclusion_tags_list = [
        'reverse'
    ]
    # Load data tags
    data_tagging = TagRawData(pathRootData)
    save_path_filtered_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData'  # parent directory where the preprocessed data should be saved to (separate folder will be created in this directory)
    dataset_tag = 'more_filtered_withlowspeed_learning_data'
    # dataset_tag = 'lookatdata_more_filtered_multilaptrustnoreverse_1'

    #______________________
    # Sample data
    sample_data = True
    # sample_data = False
    sampling_time_period = 0.1
    root_path_sampled_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'  # parent directory where the sampled data should be saved to (separate folder will be created in this directory)
    path_preprocessed_dataset = None

    #______________________
    # Preprocess data
    preprocess_data = True
    # preprocess_data = False
    path_sampled_data = None
    save_path_merged_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/LearnedModel'
    path_merged_data = None



    # ______________^^^_________________

    if tag_data:
        data_tagging = TagRawData(pathRootData)
        data_tagging.tag_log_files(overwrite=True)
        data_tagging.save_prepro_params()

    if filter_data:
        data_tagging.sort_out_data(required_data_list, required_tags_list, exclusion_tags_list)
        print('Data preprocessing started...')
        filtered_data_dict = prepare_dataset(pathRootData, data_tagging, required_data_list,
                                             start_from='20190514')

        print('Data preprocessing completed.')
        path_preprocessed_dataset = create_folder_with_time(save_path_filtered_data, dataset_tag)

        print('Now writing to file at', path_preprocessed_dataset)
        dict_keys_to_pkl(filtered_data_dict, path_preprocessed_dataset)

    if sample_data:
        print('Sampling Data...')
        path_sampled_data = sample_from_logdata(sampling_time_period, root_path_sampled_data, path_preprocessed_dataset, dataset_tag)

    if preprocess_data:
        print('Merging Data...')
        path_merged_data = merge_data(save_path_merged_data, path_sampled_data, dataset_tag, required_data_list)

        print('Calcluating disturbance on predicitons with nominal vehicle model...')
        calculate_disturbance(path_merged_data, dataset_tag)

        pass

    print('Total computing time: ', int(time.time() - t), "s")


if __name__ == "__main__":
    main()
