#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 31.05.19 08:45

@author: mvb
"""
from dataanalysisV2.data_io import create_folder_with_time, dict_keys_to_pkl
from dataanalysisV2.gokartpreprocessing.sample_from_dataset import sample_from_logdata
from dataanalysisV2.gokartpreprocessing.build_data_set import prepare_dataset
from dataanalysisV2.gokartpreprocessing.tag_raw_data import TagRawData

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
    # filter_data = True
    filter_data = False
    required_tags_list = ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
                          'MH AB [m*s^-2]', 'MH TV [rad*s^-2]']  # list of signals and tags which should be true for the logs used to build the dataset
    exclusion_tags_list = []
    # Load data tags
    data_tagging = TagRawData(pathRootData)
    save_filtered_data_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData'  # parent directory where the preprocessed data should be saved to (separate folder will be created in this directory)
    dataset_tag = 'TF_test'

    #______________________
    # Sample data
    # sample_data = True
    sample_data = False
    sampling_time_period = 0.02
    path_sampled_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'  # parent directory where the sampled data should be saved to (separate folder will be created in this directory)
    path_preprocessed_dataset = None

    #______________________
    # Preprocess data
    preprocess_data = True
    # preprocess_data = False


    # ______________^^^_________________

    if tag_data:
        data_tagging = TagRawData(pathRootData)
        data_tagging.tag_log_files(overwrite=True)
        data_tagging.save_prepro_params()

    if filter_data:
        data_tagging.sort_out_data(required_tags_list, exclusion_tags_list)
        print('Data preprocessing started...')
        filtered_data_dict = prepare_dataset(pathRootData, data_tagging, required_tags_list)
        print(type(filtered_data_dict))

        print('Data preprocessing completed.')
        path_preprocessed_dataset = create_folder_with_time(save_filtered_data_path, dataset_tag)

        print('Now writing to file at', path_preprocessed_dataset)
        dict_keys_to_pkl(filtered_data_dict, path_preprocessed_dataset)

    if sample_data:
        print('Sampling Data...')
        sample_from_logdata(sampling_time_period, path_sampled_data, path_preprocessed_dataset, dataset_tag)

    if preprocess_data:
        pass

    print('Total computing time: ', int(time.time() - t), "s")


if __name__ == "__main__":
    main()
