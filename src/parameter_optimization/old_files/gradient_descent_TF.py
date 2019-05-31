#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.05.19 15:33

@author: mvb
"""
from dataanalysisV2.data_io import getPKL
from old_files.marcsmodel_paramopt_TF import marc_vehiclemodel

import tensorflow as tf


def main():
    # Parameters
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/33333333333333/20190404T133714_01_sampledlogdata.pkl'
    batch_size = 3
    learning_rate = 0.01
    epochs = 1

    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_nrof_samples = len(dataframe)
    dataset = dataframe_to_tfdataset(dataframe)
    dataset = dataset.shuffle(total_nrof_samples, seed=None)
    batched_dataset = dataset.batch(batch_size)
    # iterator = batched_dataset.make_one_shot_iterator()
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Placeholders for tf Graph inputs
    X = tf.placeholder(tf.float32, [None, 6]) # input to Marc's model [VELX, VELY, VELROTZ, BETA, AB, TV]
    Y = tf.placeholder(tf.float32, [None, 3]) # predictions from Marc's model [ACCX, ACCY, ACCROTZ]

    # Model parameters for Marc's model
    W = tf.Variable(tf.ones([4]))
    # W = tf.Variable([9.4, 10.4, 1.08, 0.3])

    # Model
    prediction = marc_vehiclemodel(X,W)

    # Cost function, root mean square error
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, prediction))))

    # Gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        noof_batches = int(total_nrof_samples/batch_size) + 1
        for epoch in range(epochs):
            sess.run(iterator.initializer)
            for i in range(noof_batches):
                batch_X, batch_Y = sess.run(next_element)

                # plt.scatter(range(len(batch_X[:,0])),batch_X[:,0])
                pred, weights, Xval, Yval = sess.run([prediction, W, X, Y], feed_dict={X: batch_X, Y: batch_Y})
                # print('X:', batch_X, 'Y:', batch_Y)
                # print('weights', weights)
                print('X',Xval)
                print('pred', pred)
                print('measurement', Yval)

                _,c,weights = sess.run([optimizer, cost, W], feed_dict={X: batch_X, Y: batch_Y})
                print('cost:', c,'\n')

            if epoch % 10 == 0:
                print('epoch:', epoch, 'cost:', c, 'weights', weights)

        print('\n', 'Optimization finished!')
        print('epoch:', epoch, 'cost:', c, 'weights', weights)

def dataframe_to_tfdataset(dataframe):
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

    return dataset

if __name__ == '__main__':
    main()