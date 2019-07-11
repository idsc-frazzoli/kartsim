#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:25:01 2019

@author: mvb
"""

import pandas as pd
import os
import datetime
import pickle

def dataframe_from_csv(filePath):
    dataFrame = pd.read_csv(str(filePath))
    return dataFrame

def getPKL(filePath):
    dataFrame = pd.read_pickle(str(filePath))
    return dataFrame


def getDirectories(path):
    folders = [f.name for f in os.scandir(path) if f.is_dir()]
    return folders

def dict_to_csv(filePath, dict):
    with open(filePath, 'w') as f:
        for day in sorted(dict):
            f.write("%s\n"%(day))
            for log in sorted(dict[day]):
                f.write(",%s\n" % (log))
                for topic in sorted(dict[day][log]):
                    f.write(",,%s\n" % (topic))
                    try:
                        for value in sorted(dict[day][log][topic]):
                            f.write(",,,%s\n" % (value))
                    except TypeError:
                        f.write(",,,%s\n" % (dict[day][log][topic]))

def dict_from_csv(filePath):
    dict = {}
    with open(filePath, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.split(',')
            if line[0] != '':
                d0 = line[0]
                dict[d0] = {}
            elif line[1] != '':
                d1 = line[1]
                dict[d0][d1] = {}
            elif line[2] != '':
                d2 = line[2]
                dict[d0][d1][d2] = {}
            elif line[3] != '':
                d3 = line[3]
                dict[d0][d1][d2] = d3
    return dict


def create_folder_with_time(save_path, tag = None):
    current_dt = datetime.datetime.now()
    folder_name = current_dt.strftime("%Y%m%d-%H%M%S")
    if tag != None:
        folder_path = save_path + '/' + folder_name + '_' + tag
    else:
        folder_path = save_path + '/' + folder_name

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print('Error: Creating directory: ', folder_path)
        raise

    return folder_path

def dict_keys_to_pkl(dict, folder_path):
    comp_tot = len(dict)

    comp_count = 0
    for key in dict:
        filePathName = folder_path + '/' + key + '.pkl'
        try:
            with open(filePathName, 'wb') as f:
                pickle.dump(dict[key], f, pickle.HIGHEST_PROTOCOL)
            # print(key + '.pkl',' done')
            print(str(int(comp_count / comp_tot * 100)),
                  '% completed.   current file:', key + '.pkl', end='\r')
        except:
            print('Could not save ', key + '.pkl', ' to file.')
        comp_count += 1

def data_to_pkl(file_path_name, dataframe):
    try:
        with open(file_path_name, 'wb') as f:
            pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)
    except:
        print('Could not save to file at', file_path_name)
