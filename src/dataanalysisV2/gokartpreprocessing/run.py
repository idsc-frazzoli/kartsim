#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
from dataanalysisV2.dataIO import dict_from_csv, dict_to_csv, create_folder_with_time, dict_to_pkl
from dataanalysisV2.gokartpreprocessing.sortout import sort_out
from dataanalysisV2.gokartpreprocessing.buildDataSet import stirData

import time

def main():
    t = time.time()

    # ________User parameters

    pathRootData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts'  # path where all the raw logfiles are

    # preprocess data and compute inferred data from raw logs
    preprocessData = False
    requiredList = ['pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'vmu ax',
                    'vmu ay', 'pose atheta', 'MH AB', 'MH TV', 'MH BETA', ]  # list of required raw log signals
    saveDatasetPath = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData'
    datasetTag = 'test_newFormat'

    # check logs for missing or incomplete data
    sortOutData = True  # if True: checks all the raw logfiles for missing/incomplete data if not checked yet
    sortOutDataOverwrite = True  # if True: all data in preproParams-file will be overwritten
    preproParamsFileName = 'preproParams'
    preproParamsFilePath = pathRootData + '/' + preproParamsFileName + '.csv'  # file where all the information about missing/incomplete data is stored

    # ______________^^^_________________

    try:
        preproParams = dict_from_csv(preproParamsFilePath)
        print('Parameter file for preprocessing located and opened.')
    except:
        print('Parameter file for preprocessing does not exist. Creating file...')
        preproParams = {}

    if sortOutData:
        # tag logs with missing data
        preproParams = sort_out(pathRootData, preproParams, sortOutDataOverwrite)
        # save information to file
        dict_to_csv(preproParamsFilePath, preproParams)
        print('preproParams saved to ', preproParamsFilePath)

    if preprocessData:
        print('Data preprocessing started...', end='\r')
        kartDataAll = stirData(pathRootData, preproParams, requiredList)

        print('Data preprocessing completed.')
        folder_path = create_folder_with_time(saveDatasetPath, datasetTag)

        print('Now writing to file at', folder_path)
        dict_to_pkl(kartDataAll, folder_path)

    print('Total computing time: ', time.time() - t)

if __name__ == "__main__":
    main()