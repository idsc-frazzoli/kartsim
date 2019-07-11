#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 03.06.19 09:56

@author: mvb
"""
import tensorflow as tf
import numpy as np
import os
from data_visualization.data_io import create_folder_with_time

def create_batch_generator(x, y, batch_size=100, shuffle=False):
    x_copy = np.array(x)
    y_copy = np.array(y)

    nr_of_data_points, nr_of_labels = y_copy.shape

    if shuffle:
        merged_data = np.hstack((x_copy, y_copy))
        np.random.shuffle(merged_data)
        x_copy = merged_data[:,:-nr_of_labels]
        y_copy = merged_data[:,-nr_of_labels:]

    for index in range(0,nr_of_data_points, batch_size):
        yield (x_copy[index:index+batch_size, :], y_copy[index:index+batch_size,:])


def get_coeff_of_determination(labels, predictions):
    total_error = tf.reduce_sum(tf.square(tf.sub(labels, tf.reduce_mean(labels))))
    unexplained_error = tf.reduce_sum(tf.square(tf.sub(labels, predictions)))
    R_squared = tf.sub(1, tf.div(unexplained_error, total_error))
    return R_squared


class MultiLayerPerceptronFirstTry():
    def __init__(self, epochs=20, learning_rate=1e-4, shuffle=True, random_seed=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        np.random.seed(random_seed)

        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)

            self.build()

            self.init_gv = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        self.session = tf.Session(graph=g)

    def build(self):
        """builds the neural network"""

        # Placeholders for input and output data
        ph_x = tf.placeholder(shape=(None,7), dtype=tf.float32, name='ph_x')
        ph_y = tf.placeholder(shape=(None,3), dtype=tf.float32, name='ph_y')

        h1 = tf.layers.dense(inputs=ph_x, units=30, activation=tf.nn.relu, name='hidden_layer_1')

        h2 = tf.layers.dense(inputs=h1, units=30, activation=tf.nn.relu, name='hidden_layer_2')

        prediciton = tf.layers.dense(inputs=h2, units=3, activation=None)

        # Loss function
        mean_squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=ph_y, predictions=prediciton), name='mean_squared_loss')

        # Definition of the optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(mean_squared_loss, name='train_optimizer')


    def train(self, x_train, y_train, validation_set=None, init_vars=True):
        """trains the neural network"""

        # Initialize variables
        if init_vars:
            self.session.run(self.init_gv)

        self.get_train_stats(x_train)

        x_train_norm = self.normalize_data(x_train)

        # batch_generator = create_batch_generator(x_train_norm, y_train, batch_size=100, shuffle=True)
        #
        # for epoch in range(self.epochs):
        #
        #     avg_loss = 0
        #
        #     for x_batch, y_batch in batch_generator:
        #         feed_data = {'ph_x:0': x_batch, 'ph_y:0': y_batch}
        #
        #         loss, _ = self.session.run(['mean_squared_loss:0', 'train_optimizer'], feed_dict=feed_data)
        #
        #         avg_loss += loss
        #
        #     if not epoch % 5:
        #         print('Epoch', epoch, ': Average training loss:', avg_loss)
        #
        #         # if validation_set is not None:
        #             # feed_data = {'ph_x:0': x_batch, 'ph_y:0': y_batch}
        #
        #             # valid_loss, _ = self.session.run(['mean_squared_loss:0', 'train_optimizer'], feed_dict=feed_data)

    def predict(self, test_features):
        test_features_norm = self.normalize_data(test_features)
        feed_data = {'ph_x:0': test_features_norm}
        return self.session.run('prediction:0', feed_dict=feed_data)

    def get_train_stats(self, training_features):
        self.train_stats = training_features.describe()
        self.train_stats = self.train_stats.traspose()
        print(self.train_stats)

    def normalize_data(self, x):
        return (x - self.train_stats['mean']) / self.train_stats['std']

    def save_model(self, epoch, save_path='./tf_models', model_tag='mlp-model'):
        save_path_model = create_folder_with_time(save_path, model_tag)
        print('Saving NN-model to', save_path_model)
        model_path = save_path_model + '/mlp-model.ckpt'
        self.saver.save(self.session, os.path.join(save_path_model, 'mlp-model.ckpt'), global_step=epoch)

    def load_model(self, epoch, load_path):
        print('Loading NN-model from', load_path)
        self.saver.restore(self.session, os.path.join(load_path, 'mlp-model.ckpt-%d' % epoch))