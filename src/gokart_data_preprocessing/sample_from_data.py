#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:23:12 2019

@author: mvb
"""
import os
import pickle
import numpy as np
import pandas as pd
import time
import re

from config import directories
from data_visualization.mathfunction import interpolation, interpolation_w_mean
from data_visualization.data_io import create_folder_with_time, data_to_pkl
import dataIO as dio

def sample_from_logdata(sampling_time_period, path_load_data = None, dataset_tag = "test",
                        merge_data=False):
    path_save_data = os.path.join(directories['root'], 'Data', 'Sampled')
    #__user input
    if path_load_data == None:
        path_preprocessed_data = os.path.join(directories['root'], 'Data', 'Filtered')
        folders_preprocessed_data = dio.getDirectories(path_preprocessed_data)
        folders_preprocessed_data.sort()
        folders_preprocessed_data.reverse()
        for str in folders_preprocessed_data:
            if str.endswith(dataset_tag):
                defaultSim = str
                break
        # defaultSim = folders_preprocessed_data[-1]
        path_load_data = path_preprocessed_data + '/' + defaultSim

    #______^^^______
    if merge_data:
        datapool = []
    files = []
    for r, d, f in os.walk(path_load_data):
        for file in f:
            if '.pkl' in file:
                files.append([os.path.join(r, file), file])
    total_nr_of_files = len(files)

    save_folder_path = create_folder_with_time(path_save_data, dataset_tag)

    print('Sampling data from preprocessed data at', path_load_data)

    file_count = 0
    starttime = time.time()
    for file, fileName in files:
        print(int(file_count / total_nr_of_files * 100),
              '% completed.   current file:', fileName[:-4] + '_sampledlogdata.pkl   elapsed time:',
              int(time.time() - starttime), 's', end='\r')

        sampled_data = get_sampled_data(file, sampling_time_period)

        if merge_data:
            if file_count == 0:
                datapool = sampled_data
                file_count += 1
            else:
                datapool = pd.concat([datapool, sampled_data])

        else:
            filePathName = save_folder_path + '/' + fileName[:-4] + '_sampledlogdata.pkl'
            data_to_pkl(filePathName, sampled_data)
            file_count += 1

    if merge_data:
        datapool = datapool.reset_index(drop=True)
        filePathName = os.path.join(save_folder_path, 'merged_sampledlogdata.pkl')
        data_to_pkl(filePathName, datapool)

    print('Data sampling completed.')

    return save_folder_path

        
def get_sampled_data(file, samplingTimeStep):
    try:
        with open(file, 'rb') as f:
            logData = pickle.load(f)
    except:
        print('Could not open preprocessed data file at', file)
        logData = {}
    t0 = [[],[],[]]

    for topic in logData:
        t0[2].append(logData[topic][0][-1])
        t0[1].append(logData[topic][0][0])
        t0[0].append(topic)
    topicRef0 = t0[0][t0[1].index(np.max(t0[1]))]    #topic with the largest t0
    topicRef1 = t0[0][t0[2].index(np.min(t0[2]))]    #topic with the smallest t_end
    initDataFrame = True
    for topic in logData:
        # sTime, sData = interpolation_w_mean(list(logData[topic][0]), list(logData[topic][1]), logData[topicRef0][0][0], logData[topicRef1][0][-1], samplingTimeStep)
        sTime, sData = interpolation(list(logData[topic][0]), list(logData[topic][1]), logData[topicRef0][0][0], logData[topicRef1][0][-1], samplingTimeStep)
        if initDataFrame:
            dataTime = np.round(list(sTime-sTime[0]),2)
            dataSetLog = pd.DataFrame(dataTime,columns = ['time [s]'])
            initDataFrame = False
        dataSetLog.insert(len(dataSetLog.columns),topic,list(sData))

    return dataSetLog
