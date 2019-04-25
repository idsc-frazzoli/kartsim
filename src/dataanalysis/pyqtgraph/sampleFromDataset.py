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

import dataanalysis.pyqtgraph.preprocess as prep
import dataanalysis.pyqtgraph.dataIO as dio

def main():
    #__user input
    pathRootSortedData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/RawSortedData/'
    simFolders = dio.getDirectories(pathRootSortedData)
    simFolders.sort()
    defaultSim = simFolders[-1]
    pathLoadData = pathRootSortedData + '/' + defaultSim
    # pathLoadData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/RawSortedData/20190423-155633_test_MarcsModel_allData' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathSaveData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets' #path where all the raw, sorted data is that you want to sample and or batch and or split
    datasetTag = 'test_MarcsModel_allData'
    samplingTimeStep = 0.01
    
#    dataSpecificMarcsModel = True
    
    #______^^^______

    comp_tot = 0
    files = []
    for r, d, f in os.walk(pathLoadData):
        for file in f:
            if '.pkl' in file:
                files.append([os.path.join(r, file), file])
                comp_tot += 1

    currentDT = datetime.datetime.now()
    folderName = currentDT.strftime("%Y%m%d-%H%M%S")
    folderPath = pathSaveData + '/' + folderName + '_' + datasetTag

    try:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    except OSError:
        print('Error while creating directory: ', folderPath)

    print('Sampling data from preprocessed data at', pathLoadData)

    comp_count = 0
    starttime = time.time()
    for file, fileName in files:
        print(str(int(comp_count / comp_tot * 100)),
              '% completed.   current file:', fileName[:-4] + '_sampledlogdata.pkl   elapsed time:', int(time.time() - starttime), 's', end='\r')

        dataSet = getSampledData(file, samplingTimeStep)

        filePathName = folderPath + '/' + fileName[:-4] + '_sampledlogdata.pkl'

        try:
            with open(filePathName, 'wb') as f:
                pickle.dump(dataSet, f, pickle.HIGHEST_PROTOCOL)
            # print('dataset.pkl',' done')
        except:
            print('Could not save ', 'dataset.pkl' ,' to file.')

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
            dataSetLog = pd.DataFrame(list(sTime-sTime[0]),columns = ['time'])
            initDataFrame = False
        dataSetLog.insert(len(dataSetLog.columns),topic,list(sData))

    return dataSetLog


if __name__ == '__main__':
    main()