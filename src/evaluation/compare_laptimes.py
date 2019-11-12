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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def get_laptime_statistics():
    path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/old_tires/20190921', ['MPC normal warmup', 'MPC normal warmup', 'MPC normal', 'dyn_NN_0x6_None_reg0p01', 'dyn_NN_0x6_None_reg0p01_symmetric', 'kin_NN_1x16_tanh_reg0p0', 'kin_NN_1x16_softplus_reg0p0_symmetric_detailed', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed',],17.0]
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190923T161636', ['MPC normal warmup', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed', 'kin_1x16_softplus_reg0p0_symmetric_detailed',],16]
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190926T103013',['MPC normal warmup', 'MPC normal', 'kin_1x16_softplus_reg0p0_symmetric_detailed_short', 'kin_1x16_softplus_reg0p0_symmetric_detailed_short'],16]
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190926T121623', ['dyn_0x6_None_reg0p01', 'MPC normal badmap', 'MPC normal', 'dyn_0x6_None_reg0p01_symmetric', 'kin_1x16_tanh_reg0p0', 'kin_1x16_softplus_reg0p0_symmetric_detailed'],16]

    files = []
    for r, d, f in os.walk(path_w_models_a_controlpoints[0]):
        for file in f:
            if '.csv' in file and '2019' in file:
                files.append([os.path.join(r, file), file])
    files = sorted(files, key = lambda x: x[1])
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
    for index, ((file_path, file_name), model_name) in enumerate(zip(files, path_w_models_a_controlpoints[1])):
        dataframe = pd.read_csv(str(file_path), names=column_names)
        if dataframe['PATH_PROGRESS'].values[-1] > 3*path_w_models_a_controlpoints[2]:
            interp = interp1d( dataframe['PATH_PROGRESS'].values, dataframe['TIME'].values)
            start = dataframe['PATH_PROGRESS'].values[0]+path_w_models_a_controlpoints[2]
            # start = dataframe['PATH_PROGRESS'].values[0]
            stop = dataframe['PATH_PROGRESS'].values[-1] - (dataframe['PATH_PROGRESS'].values[-1] - dataframe['PATH_PROGRESS'].values[0])%17
            path_progress = np.arange(start, stop, 0.01)
            time = interp(path_progress)
            lap_time = time[int(path_w_models_a_controlpoints[2]*100):] - time[:-int(path_w_models_a_controlpoints[2]*100)]

            if dataframe['PATH_PROGRESS'].values[0] > path_w_models_a_controlpoints[2]-2.0:
                first_lap = 2
            else:
                first_lap = 1

            plt.figure(index)
            # sns.distplot(lap_time)
            plt.plot(path_progress[int(path_w_models_a_controlpoints[2]*100):]-path_w_models_a_controlpoints[2]*first_lap,lap_time)
            plt.title(file_name[-6:] + ': ' + model_name)
            # plt.ylim(9.3,11.0)

            laps = (stop - start) / 17.0
            mean_laptime = np.mean(lap_time)
            median_laptime = np.median(lap_time)
            std = np.std(lap_time)
            min = np.min(lap_time)
            max = np.max(lap_time)

            print(f'{file_name}     laps: {laps}, mean lap time: {mean_laptime}, std: {std}, min: {min}, max: {max}')
            # print((stop - start)/17.0, 'laps')
            # print(np.mean(lap_time), 's mean laptime')
            # print(np.std(lap_time), 's std')

            results_dict = {'file': file_name, 'name': model_name, 'laps': laps, 'mean_laptime': mean_laptime, 'median_laptime': median_laptime, 'std': std, 'min': min, 'max': max}

            results_model = pd.DataFrame(results_dict, index=[index])
            results = results.append(results_model)

    print(results)
    save_path = os.path.join(path_w_models_a_controlpoints[0], 'results_summary.csv')
    results.to_csv(save_path)
def get_laptimes():
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/old_tires/20190921', ['MPC normal warmup', 'MPC normal warmup', 'MPC normal', 'dyn_NN_0x6_None_reg0p01', 'dyn_NN_0x6_None_reg0p01_symmetric', 'kin_NN_1x16_tanh_reg0p0', 'kin_NN_1x16_softplus_reg0p0_symmetric_detailed', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed',],17.0, 9.8729700526528]
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190923T161636', ['MPC normal warmup', 'nomodel_1x16_tanh_reg0p0_symmetric', 'nomodel_1x16_softplus_reg0p0_symmetric_detailed', 'kin_1x16_softplus_reg0p0_symmetric_detailed',],16, 9.52415502210922]
    # path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190926T103013',['MPC normal warmup', 'MPC normal', 'kin_1x16_softplus_reg0p0_symmetric_detailed_short', 'kin_1x16_softplus_reg0p0_symmetric_detailed_short'],16]
    path_w_models_a_controlpoints = ['/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/new_tires/newtires_pathprogress/20190926T121623', ['dyn_0x6_None_reg0p01', 'MPC normal badmap', 'MPC normal', 'dyn_0x6_None_reg0p01_symmetric', 'kin_1x16_tanh_reg0p0', 'kin_1x16_softplus_reg0p0_symmetric_detailed'], 16, 9.35590322039644]

    files = []
    for r, d, f in os.walk(path_w_models_a_controlpoints[0]):
        for file in f:
            if '.csv' in file and '2019' in file:
                files.append([os.path.join(r, file), file])
    files = sorted(files, key=lambda x: x[1])
    column_names = ['TIME', 'PX', 'PY', 'HEADING', 'POSE_QUALITY', 'VX', 'VY', 'OMEGAZ', 'STEERING_SCE',
                    'BRAKE_POSITION', 'PATH_PROGRESS']

    results = pd.DataFrame()
    for index, ((file_path, file_name), model_name) in enumerate(zip(files, path_w_models_a_controlpoints[1])):
        dataframe = pd.read_csv(str(file_path), names=column_names)
        if dataframe['PATH_PROGRESS'].values[-1] > 3 * path_w_models_a_controlpoints[2]:
            interp = interp1d(dataframe['PATH_PROGRESS'].values, dataframe['TIME'].values)
            start = dataframe['PATH_PROGRESS'].values[0] + path_w_models_a_controlpoints[2]
            # start = dataframe['PATH_PROGRESS'].values[0]
            stop = dataframe['PATH_PROGRESS'].values[-1] - (
                        dataframe['PATH_PROGRESS'].values[-1] - dataframe['PATH_PROGRESS'].values[0]) % 17
            path_progress = np.arange(start, stop, 0.01)
            time = interp(path_progress)
            lap_times = time[int(path_w_models_a_controlpoints[2] * 100):] - time[:-int(
                path_w_models_a_controlpoints[2] * 100)]

            results_dict = {'relative lap time': lap_times / path_w_models_a_controlpoints[3]}
            results_model = pd.DataFrame(results_dict)
            results_model['model name'] = model_name
            results_model['tire condition'] = 'fresh'
            # print(results_model.head())
            results = results.append(results_model)
    print(results)
    # save_path = os.path.join(path_w_models_a_controlpoints[0], 'laptimes.csv')
    # results.to_csv(save_path, index=False)

def show_results():
    file_path = '/home/mvb/0_ETH/01_MasterThesis/Results/20190921_23_26_laptimes_MPC_battle/final_laptimes_forviz.csv'
    violinplot_palette = sns.color_palette('colorblind')
    # sns.set_palette(violinplot_palette[8:])
    # sns.set_palette(violinplot_palette)
    # my_pal = {"used": [0.5,0.5,0.5], "fresh": "b"}
    my_pal = violinplot_palette[8:]
    results = dataframe_from_csv(file_path)
    # print(results.head())
    sns.set(font_scale=1.2)  # crazy big
    plt.figure(figsize=(8,8))
    meanpointprops = dict(marker='D', markeredgecolor='red',
                          markerfacecolor='red')
    ax = sns.boxplot(x = 'model', y = 'relative lap time', hue="tire condition", data=results, palette=my_pal, showmeans=True, meanprops=meanpointprops, whis=5.0)
    for item in ax.get_xticklabels():
        item.set_rotation(10)

    legend_elements = [Patch(facecolor=violinplot_palette[8], edgecolor='k',
                             label='used tires'),
                       Patch(facecolor=violinplot_palette[9], edgecolor='k',
                             label='fresh tires'),
                       Line2D([0], [0], marker='D', color='w', label='average lap time',
                              markerfacecolor='red', markeredgecolor='red', markersize=7),
                       ]
    ax.legend(title='', handles=legend_elements, framealpha=1.0, facecolor='w')
    ax.set_ylabel('relative lap time ' + r'[s/s]')
    ax.set_xlabel('')
    # leg = ax.axes.get_legend()
    # new_title = 'My title'
    # leg.set_title(new_title)
    # new_labels = ['label 1', 'label 2']
    # for t, l in zip(leg.texts, new_labels): t.set_text(l)
    # print(leg.texts)
    # leg.texts = leg.texts +
    # tips = sns.load_dataset("tips")
    # print(tips.head())
    plt.tight_layout()

if __name__ == '__main__':
    # get_laptime_statistics()
    # get_laptimes()
    show_results()

