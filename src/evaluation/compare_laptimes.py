#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 24.09.19 16:56

@author: mvb
"""
import os
import pandas as pd
import numpy as np
from data_visualization.data_io import getDirectories, dataframe_from_csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

path = '/home/mvb/0_ETH/01_MasterThesis/Results/20190924_laptimes_MPC_battle_old_tires/20190921'

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append([os.path.join(r, file), file])

column_names = ['TIME','PX','PY','HEADING','POSE_QUALITY','VX','VY','OMEGAZ','STEERING_SCE','BRAKE_POSITION','PATH_PROGRESS']

# for file_path, file_name in files[0:1]:
#     print(file_name)
#     dataframe = pd.read_csv(str(file_path), names=column_names)
#     control_point_progress = dataframe['PATH_PROGRESS'].values%1
#     dcontrol_point_progress = control_point_progress[1:] - control_point_progress[:-1]
#     jump = dcontrol_point_progress<-0.8
#     crossings_below = np.zeros(len(control_point_progress)-1)
#     crossings_above = np.zeros(len(control_point_progress)-1)
#     crossings_below[jump] = control_point_progress[:-1][jump]
#     crossings_above[jump] = control_point_progress[1:][jump]
#     choose_below = 1.0 - crossings_below < crossings_above
#
#     choose_below = list(choose_below)
#
#     choose_above = jump != np.array(choose_below)
#
#     choose_above = list(choose_above)
#     choose_above.insert(0,False)
#     choose_below.insert(-1, False)
#
#     choice = [a or b for a, b in zip(choose_above, choose_below)]
#
#     control_points = dataframe.loc[choice,:]
#     control_points.loc[:,'CONTROL_POINT'] = control_points['PATH_PROGRESS'].round()
#     control_points.loc[:,'CONTROL_POINT'] = control_points['CONTROL_POINT'].values % 17
#
#     laptimes = control_points['TIME'][17:].values - control_points['TIME'][:-17].values
#     laptimes = np.insert(laptimes, 0, np.repeat(np.nan, 17))
#     control_points.loc[:,'LAP_TIMES'] = laptimes
#
#     # lap_time = dataframe['TIME'].values[17:] - dataframe['TIME'].values[:-17]
#     # # plt.plot(dataframe['PATH_PROGRESS'][:-1], dcontrol_point_progress)
#
#     # for
#     plt.plot(laptimes)

results = pd.DataFrame()

for index, (file_path, file_name) in enumerate(files):
    dataframe = pd.read_csv(str(file_path), names=column_names)
    if dataframe['PATH_PROGRESS'].values[-1] > 3*17.0:
        interp = interp1d( dataframe['PATH_PROGRESS'].values, dataframe['TIME'].values)
        start = dataframe['PATH_PROGRESS'].values[0]+17.0
        stop = dataframe['PATH_PROGRESS'].values[-1] - (dataframe['PATH_PROGRESS'].values[-1] - dataframe['PATH_PROGRESS'].values[0])%17
        path_progress = np.arange(start, stop, 0.01)
        time = interp(path_progress)
        lap_time = time[1700:] - time[:-1700]

        if dataframe['PATH_PROGRESS'].values[0] > 15.0:
            first_lap = 2
        else:
            first_lap = 1

        # plt.figure(index)
        # sns.distplot(lap_time)
        # # plt.plot(path_progress[1700:]-17.0*first_lap,lap_time)
        # plt.title(file_name)
        # # plt.ylim(9.3,11.0)

        laps = (stop - start) / 17.0
        mean_laptime = np.mean(lap_time)
        median_laptime = np.median(lap_time)
        std = np.std(lap_time)

        print(f'{file_name}     laps: {(stop - start)/17.0}, mean laptime: {np.mean(lap_time)}, std: {np.std(lap_time)}')
        # print((stop - start)/17.0, 'laps')
        # print(np.mean(lap_time), 's mean laptime')
        # print(np.std(lap_time), 's std')

        results_dict = {'name': file_name, 'laps': laps, 'mean_laptime': mean_laptime, 'median_laptime': median_laptime, 'std': std}

        results_model = pd.DataFrame(results_dict, index=[index])
        results = results.append(results_model)

print(results)
save_path = os.path.join(path, 'results_summary.csv')
results.to_csv(save_path)



