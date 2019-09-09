#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 25.06.19 18:19

@author: mvb
"""
import timeit

from data_visualization.data_io import getPKL
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import randomized_svd

random_state = 45

path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput/test_features.pkl'
path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic/test_features.pkl'
test_features = getPKL(path_data_set)
path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput/test_labels.pkl'
test_labels = getPKL(path_data_set)
path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput/train_features.pkl'
path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091159_final_data_set_kinematic/train_features.pkl'
train_features = getPKL(path_data_set)
path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/MLPDatasets/20190829-091514_final_data_set_kinematic_directinput/train_labels.pkl'
train_labels = getPKL(path_data_set)

features = train_features.append(test_features)
labels = train_labels.append(test_labels)

# data_set = features.join(labels.iloc[:,1:])
data_set = features
data_set = data_set.iloc[:,1:]
print(data_set.columns)
print(data_set.shape)

u, s, vh = randomized_svd(data_set.values.T, 10)

print(u.shape, s.shape, vh.shape)
print(s)
u = np.matrix(u[:,:6])
u_inv = u.I
variance = np.square(s)
rel_variance = variance/np.sum(variance)
plt.figure(1)
plt.bar(range(len(s)),s,color='b')
plt.figure(2)
plt.bar(range(len(rel_variance)),rel_variance,color='r')
plt.title('Singular Value Decomposition')
plt.xlabel('Components')
plt.ylabel('proportion of variance explained')

sum_rel_variance = 0
for i,rel_var in enumerate(rel_variance):
    sum_rel_variance +=rel_var*100
    print(i+1,round(sum_rel_variance,2))

sum_var = 0
vars = []
test = np.dot(u_inv,data_set.T)
for line in test:
    print(np.var(line))
    vars.append(np.var(line))
for i, var in enumerate(vars):
    sum_var += var
    print(i+1, sum_var/np.sum(vars))

# u = np.array([[2,5],[1,3]])
# u = np.matrix([[1,9,3],[4,5,6],[6,8,9]])
# u= np.matrix([[8,2,5],[7,3,1],[4,9,6]])# u = np.array([[1., 2.], [3., 4.]])
# u = np.matrix(u)
# u_inv = u.I
# print(u)
# print(u_inv)
# print(np.allclose(np.dot(u_inv, u), np.eye(len(u_inv))))

# plt.plot(dataframe['turning angle [n.a]'])
# plt.plot(test_features['turning angle [n.a]'])
# plt.show()

# print(test_features['vehicle vx [m*s^-1]'][:10])
# plt.plot(test_features['vehicle vx [m*s^-1]'][:723])
plt.show()









# # Split data_set into training and test set
# train_dataset = dataframe.sample(frac=0.8)
# train_dataset = train_dataset.reset_index(drop=True)
#
# test_dataset = dataframe.drop(train_dataset.index)
# test_dataset = test_dataset.reset_index(drop=True)
#
# train_labels = train_dataset[['disturbance vehicle ax local [m*s^-2]',
#                               'disturbance vehicle ay local [m*s^-2]',
#                               'disturbance pose atheta [rad*s^-2]']]
# train_features = train_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
#                                 'steer position cal [n.a.]', 'brake position effective [m]',
#                                 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
#
# test_labels = test_dataset[['disturbance vehicle ax local [m*s^-2]',
#                             'disturbance vehicle ay local [m*s^-2]',
#                             'disturbance pose atheta [rad*s^-2]']]
# test_features = test_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
#                               'steer position cal [n.a.]', 'brake position effective [m]',
#                               'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
# test_labels = test_labels.sort_values(by=['disturbance pose atheta [rad*s^-2]'])
# print(test_labels[['disturbance pose atheta [rad*s^-2]']])

# train_labels = train_labels.sort_values(by=['disturbance pose atheta [rad*s^-2]'])
# train_labels = train_labels[train_labels['disturbance pose atheta [rad*s^-2]'] > 100]
# train_labels = train_labels[train_labels['disturbance pose atheta [rad*s^-2]'] < -100]
# train_labels = train_labels.sort_values(by=['disturbance pose atheta [rad*s^-2]'])
# print(train_labels[['disturbance pose atheta [rad*s^-2]']])
# sns.distplot(train_labels[['disturbance pose atheta [rad*s^-2]']],bins=100);
# sns.distplot(test_labels[['disturbance pose atheta [rad*s^-2]']],bins=100);
# sns.pairplot(train_labels[['disturbance vehicle ax local [m*s^-2]',
#                             'disturbance vehicle ay local [m*s^-2]',
#                             'disturbance pose atheta [rad*s^-2]']], diag_kind="kde")
#
# sns.pairplot(test_labels[['disturbance vehicle ax local [m*s^-2]',
#                             'disturbance vehicle ay local [m*s^-2]',
#                             'disturbance pose atheta [rad*s^-2]']], diag_kind="kde")