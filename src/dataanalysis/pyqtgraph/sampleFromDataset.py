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

import dataanalysis.pyqtgraph.preprocess as prep

def main():
    #__user input
    
    pathLoadData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/RawSortedData/20190416-152207_test_MarcsModel_oneDayOnly' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathSaveData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets' #path where all the raw, sorted data is that you want to sample and or batch and or split
    datasetTag = 'MarcsModel'
    samplingTimeStep = 0.01
    
#    dataSpecificMarcsModel = True
    
    #______^^^______
    
    files = []
    for r, d, f in os.walk(pathLoadData):
        for file in f:
            if '.pkl' in file:
                files.append([os.path.join(r, file), file])
                
#    if dataSpecificMarcsModel:
#        lookupFilePath = '/home/mvb/0_ETH/01_MasterThesis/TrashFiles/forward/lookup_cur_vel_to_acc.pkl'   #file where all the information about missing/incomplete data is stored
#        try:
#            with open(lookupFilePath, 'rb') as f:
#                lookupTable = pickle.load(f)
#            print('Parameter file for preprocessing located and opened.')
#        except:
#            print('Parameter file for preprocessing does not exist. Creating file...')
#            lookupTable = pd.DataFrame()
    
    dataSet = []
    for file, fileName in files[0:1]:
        dataSetLog = getSampledData(file, samplingTimeStep)
#        dataForMarcsModel(dataSetLog, lookupTable)
        datasetName = fileName
        if len(dataSet) < 1:
            dataSet = dataSetLog
        else:
            dataSet = dataSet.append(dataSetLog)
    print(dataSet.columns)
    print(dataSet.head())
    print(dataSet.tail())
    
    currentDT = datetime.datetime.now()
    folderName = currentDT.strftime("%Y%m%d-%H%M%S")
    folderPath = pathSaveData + '/' + folderName + '_' + datasetTag
    
    try:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    except OSError:
        print('Error: Creating directory: ', folderPath)
    print('now writing to file...')
    filePathName = folderPath + '/' + datasetName[:-4] + '_samp.pkl'
    try:
        with open(filePathName, 'wb') as f:
            pickle.dump(dataSet, f, pickle.HIGHEST_PROTOCOL)
        print('dataset.pkl',' done')
    except:
        print('Could not save ', 'dataset.pkl' ,' to file.')
            
    
        
def getSampledData(file, samplingTimeStep):
    try:
        with open(file, 'rb') as f:
            logData = pickle.load(f)
        print('Parameter file for preprocessing located and opened.')
    except:
        print('Parameter file for preprocessing does not exist. Creating file...')
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
            dataSetLog = pd.DataFrame(list(sTime-sTime[0]),columns = ['time'])
            initDataFrame = False
        dataSetLog.insert(len(dataSetLog.columns),topic,list(sData))

    return dataSetLog


if __name__ == '__main__':
    main()