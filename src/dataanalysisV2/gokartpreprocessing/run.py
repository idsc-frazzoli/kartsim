#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
from dataanalysisV2.dataIO import dict_from_csv, dict_to_csv, create_folder_with_time, dict_to_pkl
from dataanalysisV2.gokartpreprocessing.sortout import sort_out
from dataanalysisV2.gokartpreprocessing.sampleFromDataset import sample_from_logdata
from dataanalysisV2.gokartpreprocessing.buildDataSet import stirData

import time

def main():
    t = time.time()

    # ________User parameters

    pathRootData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts'  # path where all the raw logfiles are

    # check logs for missing or incomplete data
    # sortOutData = True  # if True: checks all the raw logfiles for missing/incomplete data if not checked yet
    sortOutData = False
    # sortOutDataOverwrite = True  # if True: all data in preproParams-file will be overwritten
    sortOutDataOverwrite = False
    _preproParamsFileName = 'preproParams'
    preproParamsFilePath = pathRootData + '/' + _preproParamsFileName + '.csv'  # file where all the information about missing/incomplete data is stored

    # preprocess data and compute inferred data from raw logs
    preprocessData = True
    # preprocessData = False
    required_list = ['pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
                     'pose vtheta [rad*s^-1]', 'vehicle slip angle', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                     'pose atheta [rad*s^-2]', 'MH BETA [rad]', 'MH AB [m*s^-2]', 'MH TV [rad*s^-2]',
                     'multiple laps', 'high slip angles']  # list of signals and tags which should be true for the logs used to build the dataset
    nono_list = ['reverse']
    saveDatasetPath = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData' # parent directory where the preprocessed data should be saved to (separate folder will be created in this directory)
    dataset_tag = 'MM_multilap_slip_noreverse'

    # sample data
    sampleData = True
    sampling_time_period = 0.01
    path_sampled_data = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets' # parent directory where the sampled data should be saved to (separate folder will be created in this directory)


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
        print('Data preprocessing started...')
        kartDataAll = stirData(pathRootData, preproParams, required_list, nono_list)

        print('Data preprocessing completed.')
        path_preprocessed_dataset = create_folder_with_time(saveDatasetPath, dataset_tag)

        print('Now writing to file at', path_preprocessed_dataset)
        dict_to_pkl(kartDataAll, path_preprocessed_dataset)

    if sampleData:
        print('Sampling Data...')
        sample_from_logdata(sampling_time_period, path_sampled_data, path_preprocessed_dataset, dataset_tag)

    print('Total computing time: ', int(time.time() - t), "s")

if __name__ == "__main__":
    main()