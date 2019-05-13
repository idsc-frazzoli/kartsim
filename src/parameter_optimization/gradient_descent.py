#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.05.19 15:33

@author: mvb
"""
from dataanalysisV2.dataIO import getPKL
from parameter_optimization.model.marcsmodel_paramopt import marc_vehiclemodel

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Parameters
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/33333333333333/20190404T133714_01_sampledlogdata.pkl'
    batch_size = 100

    # Load and batch data
    dataframe = getPKL(dataset_path)
    dataset = dataframe_to_tfdataset(dataframe, shuffle=True)
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # Placeholders for tf Graph inputs
    X = tf.placeholder(tf.float32, [None, 6]) # input to Marc's model [VELX, VELY, VELROTZ, BETA, AB, TV]
    Y = tf.placeholder(tf.float32, [None, 3]) # predictions from Marc's model [ACCX, ACCY, ACCROTZ[0]]

    # Model parameters for Marc's model
    W = tf.Variable(tf.zeros([4]))

    # Model
    prediction = marc_vehiclemodel(X,W)

    # Cost function
    # cost = np.square(prediction - Y)

    # Gradient descent optimizer


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(5):
            batch_X, batch_Y = sess.run(next_element)
            # print(type(batch[0]))
            # print(batch[0][:,0])
            plt.scatter(range(len(batch_X[:,0])),batch_X[:,0])
        plt.show()

        # for element in batch:
        #     # print(f'features:{element[0]} labels:{element[1]}')
        #     print('__\n', element)


def dataframe_to_tfdataset(dataframe, shuffle = False):
    features = ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
                'MH AB [m*s^-2]', 'MH TV [rad*s^-2]']
    label = ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']

    dataset = (tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(dataframe[features].values, tf.float32),
                tf.cast(dataframe[label].values, tf.float32)
            )
        )
    )

    if shuffle:
        buffer_size = len(dataframe)
        dataset = dataset.shuffle(buffer_size)

    return dataset

if __name__ == '__main__':
    main()