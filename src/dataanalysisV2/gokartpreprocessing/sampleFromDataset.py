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
import datetime
import time

import showrawdata.preprocess as prep
from dataanalysisV2.dataIO import create_folder_with_time, dataframe_to_pkl
import dataIO as dio

def sample_from_logdata(sampling_time_period, path_save_data, path_load_data = None, dataset_tag = "test"):
    #__user input
    if path_load_data == None:
        path_preprocessed_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData/'
        simFolders = dio.getDirectories(path_preprocessed_data)
        simFolders.sort()
        defaultSim = simFolders[-1]
        path_load_data = path_preprocessed_data + '/' + defaultSim

    # path_load_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/RawSortedData/20190423-155633_test_MarcsModel_allData' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # path_save_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # dataset_tag = 'test_newFormat'
    # sampling_time_period = 0.01
    
#    dataSpecificMarcsModel = True
    
    #______^^^______

    comp_tot = 0
    files = []
    for r, d, f in os.walk(path_load_data):
        for file in f:
            if '.pkl' in file:
                files.append([os.path.join(r, file), file])
                comp_tot += 1

    folder_path = create_folder_with_time(path_save_data, dataset_tag)

    print('Sampling data from preprocessed data at', path_load_data)

    comp_count = 0
    starttime = time.time()
    for file, fileName in files:
        print(str(int(comp_count / comp_tot * 100)),
              '% completed.   current file:', fileName[:-4] + '_sampledlogdata.pkl   elapsed time:',
              int(time.time() - starttime), 's', end='\r')

        dataSet = getSampledData(file, sampling_time_period)

        filePathName = folder_path + '/' + fileName[:-4] + '_sampledlogdata.pkl'

        dataframe_to_pkl(filePathName, dataSet)

        comp_count += 1

    print('Data sampling completed.')

            
    
        
def getSampledData(file, samplingTimeStep):
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
        sTime, sData = prep.interpolation(list(logData[topic][0]), list(logData[topic][1]), logData[topicRef0][0][0], logData[topicRef1][0][-1], samplingTimeStep)
        if initDataFrame:
            dataTime = np.round(list(sTime-sTime[0]),2)
            dataSetLog = pd.DataFrame(dataTime,columns = ['time [s]'])
            initDataFrame = False
        dataSetLog.insert(len(dataSetLog.columns),topic,list(sData))

    return dataSetLog
