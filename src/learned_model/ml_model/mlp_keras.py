#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.06.19 13:21

@author: mvb
"""
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
# from tensorflow.keras import layers
import matplotlib.pyplot as plt

from learned_model.preprocessing.shuffle import shuffle_dataframe
from dataanalysisV2.data_io import create_folder_with_time, getDirectories, dataframe_to_pkl
import time

def coeff_of_determination(labels, predictions):
    total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labels, predictions)))
    r_squared = tf.subtract(1.0, tf.math.divide(unexplained_error, total_error))
    return r_squared

class MultiLayerPerceptron():
    def __init__(self, epochs=20, learning_rate=1e-4, batch_size=100, shuffle=True, random_seed=None, model_name='test'):
        self.root_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_models'
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.model_name = model_name
        self.model = None
        self.history = None
        self.new_model = False

        self.model_dir = None
        folder_names = getDirectories(self.root_folder)
        for name in folder_names:
            if name.endswith(model_name):
                print('Model name already exists!')
                self.model_dir = os.path.join(self.root_folder, name)

        if self.model_dir is None:
            print('Could not find model name. Creating new folder...')
            self.model_dir = create_folder_with_time(
                '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_models', model_name)
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints'))
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints','best'))
            self.new_model = True


    def load_model(self):
        try:
            load_path = os.path.join(self.model_dir, 'my_model.h5')
            self.model = tf.keras.models.load_model(load_path)
            print('Model successfully loaded from', load_path)
        except:
            print('No model found in', self.model_dir)
            raise

        try:
            self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error', coeff_of_determination])
            print('Compilation successful.')
        except:
            print('Could not compile tf model.')
            raise

    def load_checkpoint(self, checkpoint_name='best'):
        load_path = os.path.join(self.model_dir, 'model_checkpoints')
        if checkpoint_name == 'best':
            best_path = os.path.join(load_path, 'best')
            print(os.listdir(best_path))

        elif checkpoint_name == 'latest':
            checkpoint = tf.train.latest_checkpoint(load_path)
        else:
            checkpoint = tf.train.latest_checkpoint(load_path)

        # try:
        #     self.model.load_weights(checkpoint)
        # except:
        #     print('Could not load checkpoint.')
        #     raise

    def build_new_model(self):
        inputs = tf.keras.Input(shape=(7,))

        h1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
        h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
        predictions = tf.keras.layers.Dense(3, activation=None)(h2)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error', coeff_of_determination])

    def train_model(self, features, labels):
        while not self.new_model:
            answer = input('There may be an existing model. Would you like to overwrite any existing models? (y/n)')
            if answer in ('y','Y','Yes','yes'):
                break
            elif answer in ('n', 'N', 'No', 'no'):
                print('Abort: Script will be terminated.')
                sys.exit()
            else:
                continue

        if self.shuffle:
            features, labels = shuffle_dataframe(features, labels, random_seed=self.random_seed)

        self.get_train_stats(features)
        normalized_features = self.normalize_data(features)

        self.save_training_parameters()

        self.save_path_checkpoints = os.path.join(self.model_dir, 'model_checkpoints', 'mpl-{epoch:04d}.ckpt')
        self.save_path_best_checkpoint = os.path.join(self.model_dir, 'model_checkpoints', 'best', 'mpl-best.ckpt')

        # Callback: Save model checkpoints regularly
        save_checkpoints = tf.keras.callbacks.ModelCheckpoint(self.save_path_checkpoints, verbose=0, period=5)

        # Callback: Save best model checkpoint
        save_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.save_path_best_checkpoint, verbose=0,
                                                                  save_best_only=True, monitor='val_loss', mode='min')

        # Callback: Stop training if validation loss does not improve anymore
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.history = self.model.fit(normalized_features, labels, batch_size=self.batch_size, epochs=self.epochs,
                                      callbacks=[stop_early, save_checkpoints, save_best_checkpoint, PrintState()],
                                      validation_split=0.2, verbose=0)

        # Save entire model to a HDF5 file
        self.model.save(os.path.join(self.model_dir, 'my_model.h5'))
        print('Training terminated\nModel saved successfully')

    def save_training_parameters(self):
        data = np.array([self.epochs, self.learning_rate, self.batch_size, self.shuffle, self.random_seed,
                         self.model_name]).transpose()
        training_parameters = pd.DataFrame(data=data,
                                           columns=['values'],
                                           index=['epochs', 'learning_rate', 'batch_size', 'shuffle', 'random_seed',
                                                    'model_name'])

        train_params_save_path = os.path.join(self.model_dir, 'training_parameters.csv')
        training_parameters.to_csv(train_params_save_path)

        data = np.array([self.train_stats['mean'].values, self.train_stats['std'].values]).transpose()
        normalizing_parameters = pd.DataFrame(data=data,
                                              columns=['mean', 'standard deviation'],
                                              index=['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
                                                     'pose vtheta [rad*s^-1]',
                                                     'steer position cal [n.a.]', 'brake position effective [m]',
                                                     'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]'])
        norm_params_save_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
        normalizing_parameters.to_csv(norm_params_save_path)

    def save_training_history(self):
        save_path = os.path.join(self.model_dir, 'training_history')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.history is not None:
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch
            hist_save_path = os.path.join(save_path, 'history.csv')
            hist.to_csv(hist_save_path)

            plt.figure('Mean Squared Error (Loss)')
            plt.plot(hist[['epoch']].values, hist[['loss','val_loss']].values)
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.savefig(os.path.join(save_path, 'loss_mean_squ_err.pdf'))

            plt.figure('Mean Absolute Error')
            plt.plot(hist[['epoch']].values, hist[['mean_absolute_error', 'val_mean_absolute_error']].values)
            plt.legend(['Training Error', 'Validation Error'])
            plt.savefig(os.path.join(save_path, 'mean_abs_err.pdf'))

            plt.figure('Coefficient of Determination R^2')
            plt.plot(hist[['epoch']].values, hist[['coeff_of_determination', 'val_coeff_of_determination']].values)
            plt.legend(['Training', 'Validation'])
            plt.savefig(os.path.join(save_path, 'R_squared.pdf'))

            plt.show()

        else:
            print('No training history found.')
            return 0

    def show_model_summary(self):
        return self.model.summary()

    def get_train_stats(self, training_features):
        self.train_stats = training_features.describe()
        self.train_stats = self.train_stats.transpose()

    def normalize_data(self, features):
        return (features - self.train_stats['mean']) / self.train_stats['std']


class PrintState(tf.keras.callbacks.Callback):
    def __init__(self):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
            # print(logs)
            # print(type(logs))
            print('Time: {:5.1f}    Epoch: {:5.0f}    Training Loss: {:10.2f}    Validation Loss: {:10.2f}'.format(time.time() - self.t0, epoch, logs['loss'], logs['val_loss']))
        # add callbacks=[PrintDot()] to model.fit(callbacks=[.....])



