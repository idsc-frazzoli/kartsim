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
import numpy as np
import pandas as pd
from scipy.stats import kde


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
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
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

    dataframe = all_features.join(all_labels[['disturbance vehicle ax local [m*s^-2]',
       'disturbance vehicle ay local [m*s^-2]',
       'disturbance pose atheta [rad*s^-2]']])


    index_counter = 0
    for title, topics in topic_lists:
        dataframe = dataframe.sample(n=10000, random_state=16)
        # dataframe_symm = symmetry_dim_reduction(dataframe_selection)
        dataframe_symm = mirror_data(dataframe)
        dataframe_selection = dataframe_symm[topics]

        mask1 = dataframe_selection['brake position effective [m]'] < 0.025
        dataframe_selection.loc[mask1, ['brake position effective [m]']] = 0.025
        mask2 = dataframe_selection['brake position effective [m]'] > 0.05
        dataframe_selection.loc[mask2, ['brake position effective [m]']] = 0.05

        train_stats = get_train_stats(dataframe_selection)
        dataframe_symm_norm = normalize_data(train_stats, dataframe_selection)

        model = TSNE(n_components=2, random_state=0, perplexity=50, verbose=2)

        tsne_data = model.fit_transform(dataframe_symm_norm.values)

        # tsne_data = np.vstack((tsne_data.T, dataframe_selection['vehicle vx [m*s^-1]'].values[:1000])).T

        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2"))

        for topic in dataframe_symm.columns:
            plt.figure(index_counter)
            sc = plt.scatter(tsne_df['Dim_1'], tsne_df['Dim_2'], c=dataframe_symm[topic])
            plt.colorbar(sc)
            plt.title(title + ' ' + topic)
            index_counter += 1

        file_path = os.path.join(simulation_folder, f'{title}_data_set_tsne_flattened_mirroreddata_brakelim.pkl')
        data_to_pkl(file_path, tsne_df)

        file_path = os.path.join(simulation_folder, f'{title}_data_set_mirroreddata_brakelim.pkl')
        data_to_pkl(file_path, dataframe_symm)


def look_at_resulting_distributions():
    # ____________Look at results
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190820-175323_trustworthy_bigdata_vxvyfilter'
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
    simulation_file = 'state_space_data_set_1.pkl'
    simulation_file = 'state_space_data_set_mirroreddata.pkl'
    simulation_file = 'state_space_data_set_mirroreddata_brakelim.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    simulation_file = 'state_space_data_set_tsne_flattened_1.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata_brakelim.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe2 = getPKL(dataset_path)
    print(dataframe1.shape)

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
    mask1 = dataframe1['brake position effective [m]'] < 0.0
    dataframe1.loc[mask1, ['brake position effective [m]']] = 0.0
    mask2 = dataframe1['brake position effective [m]'] > 0.05
    dataframe1.loc[mask2, ['brake position effective [m]']] = 0.05
    dataframe1['brake position effective [m]'] *= 100
    dataframe1 = dataframe1.rename(columns={'brake position effective [m]': 'brake position effective [m/100]'})
    dataframe1['motor torque cmd left [A_rms]'] *= 0.001
    dataframe1 = dataframe1.rename(columns={'motor torque cmd left [A_rms]': 'motor torque cmd left [A_rms*1000]'})
    dataframe1['motor torque cmd right [A_rms]'] *= 0.001
    dataframe1 = dataframe1.rename(columns={'motor torque cmd right [A_rms]': 'motor torque cmd right [A_rms*1000]'})

    df = dataframe1.melt(var_name='groups', value_name='vals')

    sns.set(style="whitegrid")
    plt.figure(10)
    ax = sns.violinplot(x="groups", y="vals", data=df, scale='width')
    for item in ax.get_xticklabels():
        item.set_rotation(10)
    # plt.figure()
    # ax = sns.boxplot(x=dataframe1['brake position effective [m/100]'])

    # cmap = sns.cubehelix_palette(n_colors=10, start=0.5, rot=0.3, hue=1.5, as_cmap=True) #red
    cmap = sns.cubehelix_palette(n_colors=10, start=0.7, rot=0.3, hue=1.5, as_cmap=True) #orange
    # cmap = sns.cubehelix_palette(n_colors=10, start=1.6, rot=0.3, hue=1., as_cmap=True) #green
    # cmap = sns.cubehelix_palette(n_colors=10, start=2.4, rot=0.3, hue=1.5, as_cmap=True) #blue
    plt.figure(11)
    sns.kdeplot(data, shade=True, cmap=cmap ,cbar=True,shade_lowest=False)
    # sns.scatterplot(x='Dim_1', y='Dim_2', data=dataframe2, c="w", s=30, linewidth=1, marker="+")
    sns.scatterplot(x='Dim_1', y='Dim_2', data=dataframe2.sample(n=1000, random_state=42), marker="+", color='w', alpha=0.5)

    # nbins = 100
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.figure(3)
    # # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    # sc = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='coolwarm')
    # plt.contour(xi, yi, zi.reshape(xi.shape), colors='k')
    # plt.colorbar(sc)
    #
    # import scipy.stats as st
    # # Define the borders
    # deltaX = (max(x) - min(x)) / 10
    # deltaY = (max(y) - min(y)) / 10
    # xmin = min(x) - deltaX
    # xmax = max(x) + deltaX
    # ymin = min(y) - deltaY
    # ymax = max(y) + deltaY
    # print(xmin, xmax, ymin, ymax)  # Create meshgrid
    # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # values = np.vstack([x, y])
    # kernel = st.gaussian_kde(values)
    # f = np.reshape(kernel(positions).T, xx.shape)
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.gca()
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    # cset = ax.contour(xx, yy, f, colors='k')
    # plt.colorbar(cfset, ax=ax)
    # # ax.clabel(cset, inline=1, fontsize=10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.title('2D Gaussian Kernel density estimation')

def look_at_results():
    # ____________Look at results
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    # simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190820-175323_trustworthy_bigdata_vxvyfilter'
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190824-183606_trustworthy_bigdata_kinematic'
    simulation_file = 'state_space_data_set_1.pkl'
    simulation_file = 'state_space_data_set_mirroreddata.pkl'
    simulation_file = 'state_space_data_set_mirroreddata_brakelim.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    simulation_file = 'state_space_data_set_tsne_flattened_1.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata.pkl'
    simulation_file = 'state_space_data_set_tsne_flattened_mirroreddata_brakelim.pkl'
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
    for topic in disturbance:
        plt.figure(u)
        cmap = 'RdBu'

        if topic in ['vehicle vx [m*s^-1]', 'brake position effective [m]']:
            cmap = 'Reds'

        if topic in ['disturbance pose atheta [rad*s^-2]', 'disturbance vehicle ax local [m*s^-2]', 'disturbance vehicle ay local [m*s^-2]']:
            sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic].abs(), cmap=cmap)
        else:
            sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic], cmap=cmap)
        plt.title('results: ' + topic)
        if topic == 'vehicle vy [m*s^-1]':
            plt.clim(-0.6, 0.6)
        if topic == 'brake position effective [m]':
            plt.clim(0.025, 0.05)
        plt.colorbar(sc)
        # if topic == 'disturbance vehicle ay local [m*s^-2]':
        #     plt.clim(-1, 1)
        u = u + 1
    # plt.figure(20)
    # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'],
    #                  c=(dataframe1['motor torque cmd right [A_rms]']-dataframe1['motor torque cmd left [A_rms]']), cmap='RdBu')
    # plt.title('results: TV')
    # plt.colorbar(sc)
    #
    # plt.figure(21)
    # sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'],
    #                  c=((dataframe1['motor torque cmd right [A_rms]'] + dataframe1['motor torque cmd left [A_rms]'])/2.0),
    #                  cmap='RdBu')
    # plt.title('results: AB')
    # plt.colorbar(sc)

    plt.show()


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
    look_at_resulting_distributions()
