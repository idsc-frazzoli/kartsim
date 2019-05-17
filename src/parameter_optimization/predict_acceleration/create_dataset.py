#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 15.05.19 10:28

@author: mvb
"""
from dataanalysisV2.dataIO import getPKL, dataframe_to_pkl, create_folder_with_time
import numpy as np
import os
import time
import pandas as pd
np.set_printoptions(precision=4)
def main():
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/20190513-132741_MM_multilap_slip_noreverse'
    save_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization'
    tag = 'test_dataset_for_paramopt'

    file_list = []
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if '.pkl' in file:
                file_list.append([os.path.join(r, file), file])

    print('Building data set with', len(file_list), 'files from', dataset_path)

    t0 = time.time()
    dataset = collect(file_list)
    print('elapsed time:', time.time() - t0, 's')

    save_folder_path = create_folder_with_time(save_path,tag)

    file_path = save_folder_path + '/dataset.pkl'

    dataframe_to_pkl(file_path, dataset)


def collect(file_list):
    for index, [file_path, file_name] in enumerate(file_list):
        if index == 0:
            datapool = load_data(file_path)
        else:
            datapool = pd.concat([datapool, load_data(file_path)])
    # datapool = datapool.values
    return datapool

def load_data(filepath):
    dataframe = getPKL(filepath)
    if dataframe.isnull().values.any():
        print('ERROR: nan values detected in', filepath)
        for key in dataframe:
            if dataframe[key].isnull().values.any():
                print(key, 'is affected')
                print('at positions',dataframe[key][dataframe.isnull().any(axis=1)].index.values)
                print('File',filepath, 'will be skipped!')
        return pd.DataFrame()

    return dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]



if __name__ == '__main__':
    main()