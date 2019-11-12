#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 07.08.19 17:21

@author: mvb
"""

from sklearn.manifold import TSNE
from data_visualization.data_io import getPKL, data_to_pkl
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import kde
from matplotlib.ticker import EngFormatter

def main():
    topic_lists = [
        # ['vel', ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', ]],
        # ['commands', ['steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]',
        #               'motor torque cmd right [A_rms]']],
        ['state_space',
         ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]',
          'brake position effective [m]', 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']],

    ]

    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190820-175323_trustworthy_bigdata_vxvyfilter'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic'
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-092204_final_data_set_nomodel'
    simulation_file = 'test_features.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    test_features = getPKL(dataset_path)

    simulation_file = 'train_features.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    train_features = getPKL(dataset_path)

    all_features = train_features.append(test_features, ignore_index=True)

    simulation_file = 'test_labels.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    test_labels = getPKL(dataset_path)

    simulation_file = 'train_labels.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    train_labels = getPKL(dataset_path)

    all_labels = train_labels.append(test_labels, ignore_index=True)

    # dataframe = all_features.join(all_labels[['disturbance vehicle ax local [m*s^-2]',
    #    'disturbance vehicle ay local [m*s^-2]',
    #    'disturbance pose atheta [rad*s^-2]']])
    #
    dataframe = all_features.join(all_labels[['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                              'pose atheta [rad*s^-2]']])


    index_counter = 0
    for title, topics in topic_lists:
        dataframe = dataframe.sample(n=50000, random_state=16)
        # dataframe = dataframe.sample(frac=1.0, random_state=16)
        # dataframe_symm = symmetry_dim_reduction(dataframe_selection)
        dataframe_symm = mirror_data(dataframe)
        dataframe_symm = dataframe
        # dataframe_selection = dataframe_symm[topics]
        #
        # mask1 = dataframe_selection['brake position effective [m]'] < 0.025
        # dataframe_selection.loc[mask1, ['brake position effective [m]']] = 0.025
        # mask2 = dataframe_selection['brake position effective [m]'] > 0.05
        # dataframe_selection.loc[mask2, ['brake position effective [m]']] = 0.05
        #
        # train_stats = get_train_stats(dataframe_selection)
        # dataframe_symm_norm = normalize_data(train_stats, dataframe_selection)
        #
        # model = TSNE(n_components=2, random_state=0, perplexity=50, verbose=2, n_iter=5000)
        #
        # tsne_data = model.fit_transform(dataframe_symm_norm.values)
        #
        # # tsne_data = np.vstack((tsne_data.T, dataframe_selection['vehicle vx [m*s^-1]'].values[:1000])).T
        #
        # tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2"))
        #
        # for topic in dataframe_symm.columns:
        #     plt.figure(index_counter)
        #     sc = plt.scatter(tsne_df['Dim_1'], tsne_df['Dim_2'], c=dataframe_symm[topic])
        #     plt.colorbar(sc)
        #     plt.title(title + ' ' + topic)
        #     index_counter += 1
        #
        # file_path = os.path.join(simulation_folder, f'{title}_data_set_tsne_flattened.pkl')
        # data_to_pkl(file_path, tsne_df)

        file_path = os.path.join(simulation_folder, f'{title}_data_set.pkl')
        data_to_pkl(file_path, dataframe_symm)


def beautiful_signal_distributions():
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-092204_final_data_set_nomodel'
    simulation_file = 'state_space_data_set.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    print(dataframe1.shape)
    dataframe1 = dataframe1.iloc[:int(dataframe1.shape[0] / 2.0), :]
    # dataframe1 = dataframe1[dataframe1['steer position cal [n.a.]'] < 0.0]
    print(dataframe1.describe())
    mask1 = dataframe1['brake position effective [m]'] < 0.0
    dataframe1.loc[mask1, ['brake position effective [m]']] = 0.0
    # mask2 = dataframe1['brake position effective [m]'] > 0.05
    # dataframe1.loc[mask2, ['brake position effective [m]']] = 0.05
    # dataframe1['brake position effective [m]'] *= 100
    # dataframe1 = dataframe1.rename(columns={'brake position effective [m]': 'brake position effective [m/100]'})
    # dataframe1['motor torque cmd left [A_rms]'] *= 0.001
    # dataframe1 = dataframe1.rename(columns={'motor torque cmd left [A_rms]': 'motor torque cmd left [A_rms*1000]'})
    # dataframe1['motor torque cmd right [A_rms]'] *= 0.001
    # dataframe1 = dataframe1.rename(columns={'motor torque cmd right [A_rms]': 'motor torque cmd right [A_rms*1000]'})

    dataframe_vx = dataframe1[[
        'vehicle vx [m*s^-1]',
    ]]
    dataframe_vy = dataframe1[[
        'vehicle vy [m*s^-1]',
    ]]
    dataframe_vtheta = dataframe1[[
        'pose vtheta [rad*s^-1]',
    ]]
    dataframe_steer = dataframe1[[
        'steer position cal [n.a.]',
    ]]
    dataframe_brake = dataframe1[[
        'brake position effective [m]',
    ]]
    dataframe_rimo = dataframe1[[
        'motor torque cmd left [A_rms]',
        'motor torque cmd right [A_rms]',
    ]]
    dataframe_x_accel = dataframe1[[
        'vehicle ax local [m*s^-2]',
    ]]
    dataframe_y_accel = dataframe1[[
        'vehicle ay local [m*s^-2]',
    ]]
    dataframe_rot_accel = dataframe1[[
        'pose atheta [rad*s^-2]'
    ]]

    # vx_name = 'local vehicle longitudinal velocity ' + r'$[m/s]$'
    vx_name = 'longitudinal vehicle velocity ' + r'$[\mathrm{m\,s^{-1}}]$'
    vy_name = 'lateral vehicle velocity ' + r'$[\mathrm{m\,s^{-1}}]$'
    vtheta_name = 'rotational vehicle velocity ' + r'$[\mathrm{rad\,s^{-1}}]$'
    steerpos_name = 'steering wheel position ' + r'[SCE]'
    brake_name = 'brake position ' + r'$[\mathrm{m}]$'
    rimo_name = 'drive motor command ' + r'$[\mathrm{A_{RMS}}]$'
    ax_name = 'longitudinal vehicle acceleration ' + r'$[\mathrm{m\,s^{-2}}]$'
    ay_name = 'lateral vehicle acceleration ' + r'$[\mathrm{m\,s^{-2}}]$'
    atheta_name = 'rotational vehicle acceleration ' + r'$[\mathrm{rad\,s^{-2}}]$'
    df = dataframe_vx.melt(var_name='Signal', value_name='Values')

    violinplot_palette = sns.color_palette('colorblind')
    boxplot_palette = sns.color_palette('dark')
    # boxplot_palette = sns.color_palette('colorblind',desat=0)
    sns.set_palette(violinplot_palette)
    sns.set(style="whitegrid")
    sns.set(font_scale=1.2)  # crazy big

    meanpointprops = dict(marker='D', markeredgecolor='red',
                          markerfacecolor='red')

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0,gridsize=1000)
    ax.boxplot(dataframe_vx.values, whis=1.5, showfliers=False, positions=np.array([0]),
            showcaps=False,widths=0.06, patch_artist=True,
            boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
            whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
            medianprops=dict(color="w", linewidth=2 ), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set(xlabel='kernel density estimate', ylabel=vx_name)
    plt.tight_layout()

    sns.set_palette(violinplot_palette[2:])
    boxplot_palette = sns.color_palette('dark')[2:]
    # boxplot_palette = sns.dark_palette(input=violinplot_palette[2:])
    df = dataframe_vy.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_vy.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.06, patch_artist=True,
               boxprops=dict(color=[0, .3, 0], facecolor=[0, .3, 0]),
               whiskerprops=dict(color=[0, .3, 0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-1.0, 1.0)
    ax.set(xlabel='kernel density estimate', ylabel=vy_name)
    plt.tight_layout()

    sns.set_palette(violinplot_palette[3:])
    boxplot_palette = sns.color_palette('dark')[3:]
    # boxplot_palette = sns.dark_palette(input=violinplot_palette[2:])
    df = dataframe_vtheta.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_vtheta.values, whis=1.5, showfliers=False, positions=np.array([0]),
                                  showcaps=False, widths=0.06, patch_artist=True,
                                  boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
                                  whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
                                  medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-2.5, 2.5)
    ax.set(xlabel='kernel density estimate', ylabel=vtheta_name)
    plt.tight_layout()

    sns.set_palette(violinplot_palette[4:])
    boxplot_palette = sns.color_palette('dark')[4:]
    # boxplot_palette = sns.dark_palette(input=violinplot_palette[2:])
    df = dataframe_steer.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_steer.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.06, patch_artist=True,
               boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
               whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    # ax.set_ylim(-3.0, 3.0)
    ax.set(xlabel='kernel density estimate', ylabel=steerpos_name)
    plt.tight_layout()


    # sns.set_palette(violinplot_palette[2:])
    # boxplot_palette = sns.color_palette('dark')[2:]
    # # boxplot_palette = sns.dark_palette(input=violinplot_palette[2:])
    # df = dataframe_vy_vtheta_steer.melt(var_name='Signal', value_name='Values')
    # fig, ax = plt.subplots(figsize=(11, 5))
    # sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0,gridsize=1000)
    # for i in range(3):
    #     if i == 0:
    #         ax.boxplot(dataframe_vy_vtheta_steer.values[:, i], whis=1.5, showfliers=False, positions=np.array([i]),
    #                   showcaps=False, widths=0.06, patch_artist=True,
    #                   boxprops=dict(color=[0,.3,0], facecolor=[0,.3,0]),
    #                   whiskerprops=dict(color=[0,.3,0], linewidth=2),
    #                   medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    #     # ax.boxplot(dataframe_vy_vtheta_steer.values[:, i], whis=1.5, showfliers=False, positions=np.array([i]),
    #     #            showcaps=False, widths=0.06, patch_artist=True,
    #     #            boxprops=dict(color=[0.9,0.9,0.9], facecolor=[0.9,0.9,0.9]),
    #     #            whiskerprops=dict(color=[0.9,0.9,0.9], linewidth=2),
    #     #            medianprops=dict(color="k", linewidth=2))
    #     else:
    #         ax.boxplot(dataframe_vy_vtheta_steer.values[:,i], whis=1.5, showfliers=False, positions=np.array([i]),
    #                    showcaps=False, widths=0.06, patch_artist=True,
    #                    boxprops=dict(color=boxplot_palette[i], facecolor=boxplot_palette[i]),
    #                    whiskerprops=dict(color=boxplot_palette[i], linewidth=2),
    #                    medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    #
    # plt.xticks([0,1,2], [vy_name,vtheta_name,steerpos_name])
    # # for item in ax.get_xticklabels():
    # #     item.set_rotation(10)
    # ax.set_xlim(-0.5,2.5)
    # ax.set_ylim(-2.0, 2.0)
    # ax.set(xlabel='kernel density estimate', ylabel='values')
    # plt.tight_layout()


    sns.set_palette(violinplot_palette[5:])
    boxplot_palette = sns.color_palette('dark')[5:]
    df = dataframe_brake.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_brake.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.02, patch_artist=True,
               boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
               whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-0.005, 0.05)
    ax.set(xlabel='kernel density estimate', ylabel=brake_name)
    plt.tight_layout()


    sns.set_palette(violinplot_palette[6:])
    boxplot_palette = sns.color_palette('dark')[6:]
    # boxplot_palette = sns.dark_palette(input=violinplot_palette[2:])
    df = dataframe_rimo.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    for i in range(2):
        bp = ax.boxplot(dataframe_rimo.values[:, i], whis=1.5, showfliers=False, positions=np.array([i]),
                   showcaps=False, widths=0.06, patch_artist=True,
                   boxprops=dict(color=boxplot_palette[i], facecolor=boxplot_palette[i]),
                   whiskerprops=dict(color=boxplot_palette[i], linewidth=2),
                   medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)

    plt.setp(bp['means'], color='m')
    plt.xticks([0, 1, 2], ['left motor', 'right motor'])
    # for item in ax.get_xticklabels():
    #     item.set_rotation(10)
    ax.set_xlim(-0.5, 1.5)
    ax.set(xlabel='kernel density estimate', ylabel=rimo_name)
    plt.tight_layout()


    sns.set_palette(violinplot_palette[8:])
    boxplot_palette = sns.color_palette('dark')[8:]
    df = dataframe_x_accel.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_x_accel.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.06, patch_artist=True,
               boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
               whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-4.0, 2.6)
    ax.set(xlabel='kernel density estimate', ylabel=ax_name)
    plt.tight_layout()

    sns.set_palette(violinplot_palette[9:])
    boxplot_palette = sns.color_palette('dark')[9:]
    df = dataframe_y_accel.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_y_accel.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.06, patch_artist=True,
               boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
               whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-1.5, 1.5)
    ax.set(xlabel='kernel density estimate', ylabel=ay_name)
    plt.tight_layout()

    sns.set_palette(violinplot_palette[1:])
    boxplot_palette = sns.color_palette('dark')[1:]
    df = dataframe_rot_accel.melt(var_name='Signal', value_name='Values')
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x="Signal", y="Values", data=df, scale='width', inner=None, linewidth=0, gridsize=1000)
    ax.boxplot(dataframe_rot_accel.values, whis=1.5, showfliers=False, positions=np.array([0]),
               showcaps=False, widths=0.06, patch_artist=True,
               boxprops=dict(color=boxplot_palette[0], facecolor=boxplot_palette[0]),
               whiskerprops=dict(color=boxplot_palette[0], linewidth=2),
               medianprops=dict(color="w", linewidth=2), showmeans=True, meanprops=meanpointprops)
    ax.set(xticklabels=[])
    ax.set_ylim(-2.5, 2.5)
    ax.set(xlabel='kernel density estimate', ylabel=atheta_name)
    plt.tight_layout()

def look_at_resulting_distributions():
    # ____________Look at results
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190820-175323_trustworthy_bigdata_vxvyfilter'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic'
    # simulation_file = 'state_space_data_set_1.pkl'
    # simulation_file = 'state_space_data_set_mirroreddata.pkl'
    # simulation_file = 'state_space_data_set_mirroreddata_brakelim.pkl'
    simulation_file = 'state_space_data_set.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    print(dataframe1.shape)
    # dataframe1 = dataframe1.iloc[:int(dataframe1.shape[0]/2.0),:]
    # simulation_file = 'state_space_data_set_tsne_flattened_1.pkl'
    # simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata.pkl'
    # simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata_brakelim.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe2 = getPKL(dataset_path)
    # dataframe2 = dataframe2.sample(n=5000, random_state=42)

    data = dataframe2.values
    x, y = data.T
    dataframe1 = dataframe1[[
        'vehicle vx [m*s^-1]',
        'vehicle vy [m*s^-1]',
        'pose vtheta [rad*s^-1]',
        'steer position cal [n.a.]',
        'brake position effective [m]',
        'motor torque cmd left [A_rms]',
        'motor torque cmd right [A_rms]',
    ]]
    # mask1 = dataframe1['brake position effective [m]'] < 0.0
    # dataframe1.loc[mask1, ['brake position effective [m]']] = 0.0
    # mask2 = dataframe1['brake position effective [m]'] > 0.05
    # dataframe1.loc[mask2, ['brake position effective [m]']] = 0.05
    # dataframe1['brake position effective [m]'] *= 100
    # dataframe1 = dataframe1.rename(columns={'brake position effective [m]': 'brake position effective [m/100]'})
    # dataframe1['motor torque cmd left [A_rms]'] *= 0.001
    # dataframe1 = dataframe1.rename(columns={'motor torque cmd left [A_rms]': 'motor torque cmd left [A_rms*1000]'})
    # dataframe1['motor torque cmd right [A_rms]'] *= 0.001
    # dataframe1 = dataframe1.rename(columns={'motor torque cmd right [A_rms]': 'motor torque cmd right [A_rms*1000]'})

    df = dataframe1.melt(var_name='Signal', value_name='Value')

    # sns.set(style="whitegrid")
    # sns.set(font_scale=1.5)  # crazy big
    # plt.figure(10, figsize=(14,7))
    # ax = sns.violinplot(x="Signal", y="Value", data=df, scale='width')
    # for item in ax.get_xticklabels():
    #     item.set_rotation(10)
    # # plt.figure()
    # # ax = sns.boxplot(x=dataframe1['brake position effective [m/100]'])

    # sns.set(style="whitegrid")
    sns.set(font_scale=1.1)  # crazy big
    # sns.set(font_size=10)  # crazy big
    # cmap = sns.cubehelix_palette(n_colors=10, start=0.5, rot=0.3, hue=1.5, as_cmap=True) #red
    cmap = sns.cubehelix_palette(n_colors=10, start=0.7, rot=0.3, hue=1.5, as_cmap=True) #orange
    # cmap = sns.cubehelix_palette(n_colors=10, start=1.6, rot=0.3, hue=1., as_cmap=True) #green
    # cmap = sns.cubehelix_palette(n_colors=10, start=2.4, rot=0.3, hue=1.5, as_cmap=True) #blue
    plt.figure(11,figsize=(7,5))
    # ax = sns.kdeplot(data, shade=True, cmap=cmap ,cbar=False,shade_lowest=False)
    ax = sns.kdeplot(data, shade=True, cmap=cmap ,cbar=True,shade_lowest=False, cbar_kws={'format': '%1.1E'})
    # sns.scatterplot(x='Dim_1', y='Dim_2', data=dataframe2, color="w", marker="+")
    ax.xaxis.set_label_text("dimension 1")
    ax.yaxis.set_label_text("dimension 2")
    plt.axis('equal')

    # ax.xaxis.set_label_text("")
    # ax.yaxis.set_label_text("")
    # # sns.scatterplot(x='Dim_1', y='Dim_2', data=dataframe2.sample(n=1000, random_state=42), marker="+", color='w', alpha=0.3)
    # from matplotlib.patches import Patch, Polygon
    # from matplotlib.collections import PatchCollection
    # from matplotlib.path import Path
    # polygon_list = [
    #     [((38,-44),(15, -50),(14,-118),(111,-128),(130,-102),(75,-88),(60,-30)),100,'braking'],
    #     [((14, -118),(111, -128), (48, -200), (-3, -131 )), 70,'fast forward driving'],
    #     [((14, -118),(15, -50),(-150, -84.5),(-150, -162),(-60, -162),(-3, -131 )), 60, 'TV left'],
    #     [((-60, -162),(-3, -131),(48, -200),(-30, -210)), 50,'drift left'],
    #     [((130,-102),(75,-88),(60,-30),(116, -55),(160, -78)), 50,'drift right'],
    #     [((60,-30), (110, 0), (167, 44), (213, -10), (200, -61),(160, -78),(116, -55)), 60, 'TV right'],
    #     [((167, 44),(110, 0),(60,-30),(38,-44),(15, -50),(-150, -84.5),(-185, 25),(-81, 175),(135, 175)), 0, 'slow driving'],
    # ]
    # patches = []
    # colors = []
    # for points,color,name in polygon_list:
    #     polygon = Polygon(points, True)
    #     patches.append(polygon)
    #     colors.append(color)
    # p = PatchCollection(patches, alpha=0.4, cmap=plt.cm.jet)
    # p.set_array(np.array(colors))
    # ax.add_collection(p)
    # plt.axis('off')
    # plt.axis('equal')
    # # plt.colorbar(p, ax=ax)
    # legend_elements = [
    #     Patch(color='blue', alpha=0.4, label='slow driving'),
    #     Patch(color='yellow', alpha=0.4, label='high torque vectoring'),
    #     Patch(color='orange', alpha=0.4, label='high forward acceleration'),
    #     Patch(color='red', alpha=0.4, label='braking'),
    #     Patch(color='green', alpha=0.4, label='high slip angels'),
    # ]
    #
    # # Create the figure
    # ax.legend(handles=legend_elements, loc='top', bbox_to_anchor=(0.55, 0, 0.5, 1), facecolor='w', fontsize=14)
    # # ax.set_xlim(-100,100)
    # for points,color,name in polygon_list:
    #     poly_path = Path(points)
    #     point_is_contained = poly_path.contains_points(dataframe2.values)
    #     print(name, np.sum(point_is_contained)/len(dataframe2)*100, '%')
    # plt.tight_layout()
    #
    # # nbins = 100
    # # k = kde.gaussian_kde(data.T)
    # # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # # plt.figure(3)
    # # # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    # # sc = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='coolwarm')
    # # plt.contour(xi, yi, zi.reshape(xi.shape), colors='k')
    # # plt.colorbar(sc)
    # #
    # # import scipy.stats as st
    # # # Define the borders
    # # deltaX = (max(x) - min(x)) / 10
    # # deltaY = (max(y) - min(y)) / 10
    # # xmin = min(x) - deltaX
    # # xmax = max(x) + deltaX
    # # ymin = min(y) - deltaY
    # # ymax = max(y) + deltaY
    # # print(xmin, xmax, ymin, ymax)  # Create meshgrid
    # # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # # positions = np.vstack([xx.ravel(), yy.ravel()])
    # # values = np.vstack([x, y])
    # # kernel = st.gaussian_kde(values)
    # # f = np.reshape(kernel(positions).T, xx.shape)
    # # fig = plt.figure(figsize=(8, 8))
    # # ax = fig.gca()
    # # ax.set_xlim(xmin, xmax)
    # # ax.set_ylim(ymin, ymax)
    # # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    # # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    # # cset = ax.contour(xx, yy, f, colors='k')
    # # plt.colorbar(cfset, ax=ax)
    # # # ax.clabel(cset, inline=1, fontsize=10)
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # plt.title('2D Gaussian Kernel density estimation')

def show_individual_distributions():
    # ____________Look at results
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic'

    simulation_file = 'state_space_data_set.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)

    simulation_file = 'state_space_data_set_tsne_flattened.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe2 = getPKL(dataset_path)

    dataframe1 = dataframe1[[
        'vehicle vx [m*s^-1]',
        'vehicle vy [m*s^-1]',
        'pose vtheta [rad*s^-1]',
        'steer position cal [n.a.]',
        'brake position effective [m]',
        'motor torque cmd left [A_rms]',
        'motor torque cmd right [A_rms]',
    ]]

    vx_name = 'longitudinal vehicle velocity ', r'$[\mathrm{m\,s^{-1}}]$'
    vy_name = 'lateral vehicle velocity ', r'$[\mathrm{m\,s^{-1}}]$'
    vtheta_name = 'rotational vehicle velocity ', r'$[\mathrm{rad\,s^{-1}}]$'
    steerpos_name = 'steering wheel position ', r'[SCE]'
    brake_name = 'brake position ', r'$[m]$'
    rimol_name = 'drive motor command left ', r'$[\mathrm{A_{RMS}}]$'
    rimor_name = 'drive motor command right ', r'$[\mathrm{A_{RMS}}]$'
    slipangle_name = 'slip angle ', r'[rad]'
    grid_x = np.arange(-165, 195, 1)
    grid_y = np.arange(-200, 181, 1)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    from scipy.interpolate import griddata
    points = dataframe2.values
    slip_angle = np.arctan2(dataframe1['vehicle vy [m*s^-1]']-0.46*dataframe1['pose vtheta [rad*s^-1]'],dataframe1['vehicle vx [m*s^-1]'])
    slip_angle[(slip_angle.abs()>0.1) & (dataframe1['vehicle vx [m*s^-1]']<2.0)] = 0.0
    list = [
        [dataframe1['vehicle vx [m*s^-1]'].values, np.arange(0, 11, 1), vx_name],
        [dataframe1['vehicle vy [m*s^-1]'].values, np.arange(-0.5, 0.6, 0.1), vy_name],
        # [dataframe1['pose vtheta [rad*s^-1]'].values, np.arange(-1.5, 1.8, 0.3), vtheta_name],
        # [dataframe1['steer position cal [n.a.]'].values, 10, steerpos_name],
        # [dataframe1['brake position effective [m]'].values, np.arange(0.025, 0.0525, 0.0025), brake_name],
        # [dataframe1['motor torque cmd left [A_rms]'].values, 10, rimol_name],
        # [dataframe1['motor torque cmd right [A_rms]'].values, 10, rimor_name],
        # [slip_angle, np.arange(-0.1, 0.11, 0.01), slipangle_name],
    ]

    sns.set(font_scale=1.1, style='white')  # crazy big
    for i, (values, range, name) in enumerate(list):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
        plt.figure(i,figsize=(7,5))
        # plt.subplot(121)
        # # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=values, cmap='RdBu')
        # if 'lateral' in name[0]:
        #     sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=values, cmap='RdBu')
        #     plt.clim(-0.2, 0.2)
        # elif 'slip' in name[0]:
        #     sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=values, cmap='RdBu')
        #     plt.clim(-0.1, 0.1)
        # elif 'longitudinal' in name[0]:
        #     sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=values, cmap='coolwarm')
        #     cbar = plt.colorbar(sc)
        # else:
        #     sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=values, cmap='RdBu')
        # plt.subplot(122)
        if 'brake' in name or 'longitudinal' in name:
            sc1 = plt.contourf(grid_x, grid_y, grid_z1, range, extend='both')
        else:
            sc1 = plt.contourf(grid_x, grid_y, grid_z1, range, cmap='RdBu', extend='both')
        ax = sns.scatterplot(x='Dim_1', y='Dim_2', data=dataframe2.sample(n=2000, random_state=42), marker="+",
                             color='gray', alpha=0.5)
        cbar = plt.colorbar(sc1)
        plt.title(name[0])
        ax.xaxis.set_label_text("dimension 1")
        ax.yaxis.set_label_text("dimension 2")
        cbar.ax.set_ylabel(name[1], rotation=0, size=14, x=0., y=1.05)
        plt.axis('equal')
    plt.show()

def look_at_results():
    # ____________Look at results
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190820-175323_trustworthy_bigdata_vxvyfilter'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic'
    # simulation_file = 'state_space_data_set_1.pkl'
    # simulation_file = 'state_space_data_set_mirroreddata.pkl'
    # simulation_file = 'state_space_data_set_mirroreddata_brakelim.pkl'
    simulation_file = 'state_space_data_set.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    # simulation_file = 'state_space_data_set_tsne_flattened_1.pkl'
    # simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata.pkl'
    # simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata_brakelim.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe2 = getPKL(dataset_path)

    topics = [
        # 'vehicle vx [m*s^-1]',
        # 'vehicle vy [m*s^-1]',
        # 'pose vtheta [rad*s^-1]',
        # 'steer position cal [n.a.]',
        # 'brake position effective [m]',
        # 'motor torque cmd left [A_rms]',
        # 'motor torque cmd right [A_rms]',
    ]
    disturbance = [
        'vehicle vx [m*s^-1]',
        'vehicle vy [m*s^-1]',
        'pose vtheta [rad*s^-1]',
        'steer position cal [n.a.]',
        'brake position effective [m]',
        'motor torque cmd left [A_rms]',
        'motor torque cmd right [A_rms]',
        # 'disturbance vehicle ax local [m*s^-2]',
        # 'disturbance vehicle ay local [m*s^-2]',
        # 'disturbance pose atheta [rad*s^-2]',
    ]
    u = 0
    for topic in topics:
        plt.figure(u)
        sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic], cmap='RdBu')
        plt.title('results: ' + topic)
        plt.colorbar(sc)
        if topic == 'vehicle vy [m*s^-1]':
            plt.clim(-0.2, 0.2)
        # if topic == 'vehicle vx [m*s^-1]':
        #     plt.clim(0, 0.1)
        u = u + 1
    dataframe1 = dataframe1.reset_index(drop=True)
    split = 3
    # dataframe2 = dataframe2[dataframe1['disturbance vehicle ay local [m*s^-2]'].abs() > split]
    # dataframe1 = dataframe1[dataframe1['disturbance vehicle ay local [m*s^-2]'].abs() > split]
    slip_angle = np.arctan2(dataframe1['vehicle vy [m*s^-1]']-0.46*dataframe1['pose vtheta [rad*s^-1]'],dataframe1['vehicle vx [m*s^-1]'])
    slip_angle[(slip_angle.abs()>0.1) & (dataframe1['vehicle vx [m*s^-1]']<2.0)] = 0.0

    print('slip_angle', slip_angle[(slip_angle.abs()>0.09)].count())
    print('vx > 4m/s', dataframe2[dataframe1['vehicle vx [m*s^-1]']>5.0].count())
    print('all data', dataframe2.count())
    print('vy large neg', dataframe2[(dataframe2['Dim_1'].between(-20.0, 7.0)) & (dataframe2['Dim_2'].between(-185.0, -153.0))].count())
    print('vy large pos', dataframe2[(dataframe2['Dim_1'].between(88, 120)) & (dataframe2['Dim_2'].between(-93, -58))].count())
    print('braking', dataframe2[dataframe1['brake position effective [m]']>0.03].count())
    print('high TV l', dataframe2[(dataframe2['Dim_1'].between(-200, -30)) & (dataframe2['Dim_2'].between(-200, -84))].count())
    print('high TV r', dataframe2[(dataframe2['Dim_1'].between(82, 200)) & (dataframe2['Dim_2'].between(-56, 4))].count())
    print('high speed/forward acc', dataframe2[(dataframe2['Dim_1'].between(3, 86)) & (dataframe2['Dim_2'].between(-164, -120))].count() +
          dataframe2[(dataframe2['Dim_1'].between(17, 200)) & (dataframe2['Dim_2'].between(-200, -164))].count())
    print('slow driving', dataframe2[(dataframe2['Dim_2'].between(4, 200))].count())
    print('reverse acc motor', dataframe2[(dataframe2['Dim_1'].between(-71, -68)) & (dataframe2['Dim_2'].between(162, 165))].count())

    dataframe1 = dataframe1.iloc[:int(dataframe1.shape[0] / 2.0), :]
    # print('vy', np.max(dataframe1['vehicle vy [m*s^-1]'].values))
    # print('vy', np.min(dataframe1['vehicle vy [m*s^-1]'].values))
    # print('vtheta', np.max(dataframe1['pose vtheta [rad*s^-1]'].values))
    # print('vtheta', np.max(dataframe1['pose vtheta [rad*s^-1]'].values))
    # print('motortorque l', np.max(dataframe1['motor torque cmd left [A_rms]'].values))
    # print('motortorque l', np.min(dataframe1['motor torque cmd left [A_rms]'].values))
    # print('motortorque r', np.max(dataframe1['motor torque cmd right [A_rms]'].values))
    # print('motortorque r', np.min(dataframe1['motor torque cmd right [A_rms]'].values))

    # for topic in disturbance:
    #     plt.figure(u)
    #     cmap = 'RdBu'
    #
    #     if topic in [
    #         # 'vehicle vx [m*s^-1]',
    #         'brake position effective [m]']:
    #         cmap = 'Reds'
    #
    #     if topic in ['disturbance pose atheta [rad*s^-2]', 'disturbance vehicle ax local [m*s^-2]', 'disturbance vehicle ay local [m*s^-2]']:
    #         sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic].abs(), cmap=cmap)
    #     else:
    #         sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic], cmap=cmap)
    #     plt.title('results: ' + topic)
    #     if topic == 'vehicle vy [m*s^-1]':
    #         plt.clim(-0.6, 0.6)
    #     if topic == 'vehicle vx [m*s^-1]':
    #         plt.clim(0, 10)
    #     if topic == 'brake position effective [m]':
    #         plt.clim(0.025, 0.05)
    #     plt.colorbar(sc)
    #     # if topic == 'disturbance vehicle ay local [m*s^-2]':
    #     #     plt.clim(-1, 1)
    #     u = u + 1
    #     # plt.vlines(x=[-20,7,88,120], ymin=-190, ymax=175)
    #     # plt.hlines(y=[-185,-153,-93,-58], xmin=-190, xmax=175)
    # plt.figure(50)
    # cmap = 'RdBu'
    # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=slip_angle, cmap=cmap)
    # plt.title('results: ' + 'slip_angle')
    # plt.clim(-0.174532925, 0.174532925)
    # plt.colorbar(sc)
    #
    # # plt.figure(20)
    # # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'],
    # #                  c=(dataframe1['motor torque cmd right [A_rms]']-dataframe1['motor torque cmd left [A_rms]']), cmap='RdBu')
    # # plt.title('results: TV')
    # # plt.colorbar(sc)
    # #
    # # plt.figure(21)
    # # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'],
    # #                  c=((dataframe1['motor torque cmd right [A_rms]'] + dataframe1['motor torque cmd left [A_rms]'])/2.0),
    # #                  cmap='RdBu')
    # # plt.title('results: AB')
    # # plt.colorbar(sc)
    #
    # plt.show()


def get_train_stats(training_features):
    train_stats = training_features.describe()
    train_stats = train_stats.transpose()
    return train_stats


def normalize_data(train_stats, features):
    return (features - train_stats['mean']) / train_stats['std']

def mirror_data(df_features):
    features = df_features.iloc[:,1:].copy()
    mask = np.array([1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0])
    symm_features = features.multiply(mask)
    symm_features.loc[:, ['motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']] = symm_features.loc[:, ['motor torque cmd right [A_rms]', 'motor torque cmd left [A_rms]']].values
    features = features.append(symm_features)
    return features

def symmetry_dim_reduction(df_features):
    features = df_features.copy()
    steer_less_zero = features['steer position cal [n.a.]'] < 0
    mirror_vel_steer = steer_less_zero * -2 + 1
    features[['vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]']] = features[
        ['vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]']].mul(mirror_vel_steer, axis=0)
    features.loc[steer_less_zero, ['motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']] = features.loc[
        steer_less_zero, ['motor torque cmd right [A_rms]', 'motor torque cmd left [A_rms]']].values
    return features

def look_at_data_set():
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    simulation_file = 'merged_sampledlogdata.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe = getPKL(dataset_path)
    print(dataframe['vehicle vx [m*s^-1]'].describe())

if __name__ == '__main__':
    # main()
    # look_at_data_set()
    # look_at_results()
    # look_at_resulting_distributions()
    beautiful_signal_distributions()
    # show_individual_distributions()
