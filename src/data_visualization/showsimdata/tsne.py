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


def main():
    topic_lists = [
        # ['vel', ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', ]],
        # ['commands', ['steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]',
        #               'motor torque cmd right [A_rms]']],
        ['state_space',
         ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]',
          'brake position effective [m]', 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']],

    ]

    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190813-141626_trustworthy_bigdata'
    simulation_file = 'test_features.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    test_features = getPKL(dataset_path)

    simulation_file = 'train_features.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    train_features = getPKL(dataset_path)

    all_features = train_features.append(test_features, ignore_index=True)
    dataframe = all_features

    index_counter = 0
    for title, topics in topic_lists:
        dataframe = dataframe.sample(n=10000, random_state=42)
        dataframe_selection = dataframe[topics]
        dataframe_symm = symmetry_dim_reduction(dataframe_selection)
        train_stats = get_train_stats(dataframe_symm)
        dataframe_symm_norm = normalize_data(train_stats, dataframe_symm)

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
    #
    #     file_path = os.path.join(simulation_folder, f'{title}_data_set_tsne_flattened_1.pkl')
    #     data_to_pkl(file_path, tsne_df)
    #
    #     file_path = os.path.join(simulation_folder, f'{title}_data_set_1.pkl')
    #     data_to_pkl(file_path, dataframe)


def look_at_results():
    # ____________Look at results
    simulation_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190809-181131_trustworthy_bigdata_merged'
    simulation_file = 'state_space_data_set_1.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe1 = getPKL(dataset_path)
    simulation_file = 'state_space_data_set_tsne_flattened_1.pkl'
    dataset_path = os.path.join(simulation_folder, simulation_file)
    dataframe2 = getPKL(dataset_path)

    topics = ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]',
              'brake position effective [m]', 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']
    u = 0
    for topic in topics:
        plt.figure(u)
        sc = plt.scatter(dataframe2['Dim_1'], dataframe2['Dim_2'], c=dataframe1[topic])
        plt.title('results: ' + topic)
        plt.colorbar(sc)
        u = u + 1
    plt.show()


def get_train_stats(training_features):
    train_stats = training_features.describe()
    train_stats = train_stats.transpose()
    return train_stats


def normalize_data(train_stats, features):
    return (features - train_stats['mean']) / train_stats['std']


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
    main()
    # look_at_data_set()
    # look_at_results()
