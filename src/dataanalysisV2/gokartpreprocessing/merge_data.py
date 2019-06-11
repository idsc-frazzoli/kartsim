#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 15.05.19 10:28

@author: mvb
"""
from dataanalysisV2.data_io import getPKL, dataframe_to_pkl, create_folder_with_time, getDirectories
import numpy as np
import os
import time
import pandas as pd
np.set_printoptions(precision=4)

def merge_data(save_path_merged_data, load_path_sampled_data=None, tag='test', required_data_list=None):

    if load_path_sampled_data == None:
        path_preprocessed_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'
        folders_preprocessed_data = getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(tag):
                defaultSim = str
                break
        load_path_sampled_data = path_preprocessed_data + '/' + defaultSim

    file_list = []
    for r, d, f in os.walk(load_path_sampled_data):
        for file in f:
            if '.pkl' in file:
                file_list.append([os.path.join(r, file), file])

    print('Building data set with', len(file_list), 'files from', load_path_sampled_data)

    t0 = time.time()
    dataset = collect(file_list, required_data_list)
    print('elapsed time:', time.time() - t0, 's')

    save_folder_path = create_folder_with_time(save_path_merged_data,tag)

    file_path = save_folder_path + '/dataset.pkl'

    dataframe_to_pkl(file_path, dataset)

    print('Data set saved to', file_path)

    return save_folder_path


def collect(file_list, required_data_list):
    for index, [file_path, file_name] in enumerate(file_list):
        if index == 0:
            datapool = load_data(file_path, required_data_list)
        else:
            datapool = pd.concat([datapool, load_data(file_path, required_data_list)])
    # datapool = datapool.values
    return datapool

def load_data(filepath, required_data_list):
    dataframe = getPKL(filepath)
    if dataframe.isnull().values.any():
        print('ERROR: nan values detected in', filepath)
        for key in dataframe:
            if dataframe[key].isnull().values.any():
                print(key, 'is affected')
                print('at positions',dataframe[key][dataframe.isnull().any(axis=1)].index.values)
                print('File',filepath, 'will be skipped!')
        return pd.DataFrame()
    get_these_signals = ['time [s]'] + required_data_list
    return dataframe[get_these_signals]
