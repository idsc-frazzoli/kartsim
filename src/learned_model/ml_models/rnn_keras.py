#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.07.19 09:25

@author: mvb
"""
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from gokart_data_preprocessing.shuffle import shuffle_list
from data_visualization.data_io import create_folder_with_time, getDirectories
import time
import sys
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class LongShortTermMemoryNetwork():
    def __init__(self, epochs=20, learning_rate=1e-3, decay=1e-6, batch_size=100, input_sequence_length=5, time_step = 0.1, shuffle=True, random_seed=None, model_name='test',
                 predict_only=False):
        self.root_folder = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/learned_model/trained_rnn_models'
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = decay
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.model_name = model_name
        self.input_sequence_length = input_sequence_length
        self.time_step = time_step
        self.model = None
        self.history = None
        self.new_model = False
        self.means = np.array([])
        self.stds = np.array([])
        self.graph = None
        self.sess = tf.Session()

        if predict_only:
            tf.keras.backend.set_learning_phase(0)

        self.model_dir = 'no model path'
        folder_names = getDirectories(self.root_folder)
        for name in folder_names:
            if name.endswith(model_name):
                print('Model name already exists!')
                self.model_dir = os.path.join(self.root_folder, name)
                print('modeldir', self.model_dir)

        if self.model_dir is 'no model path':
            # print('New model name! Creating new folder...')
            self.model_dir = create_folder_with_time(self.root_folder, self.model_name)
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints'))
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints', 'best'))
            self.new_model = True

    def load_model(self):
        tf.keras.backend.set_session(self.sess)
        try:
            load_path = os.path.join(self.model_dir, 'my_model.h5')
            self.model = tf.keras.models.load_model(load_path, custom_objects={
                'coeff_of_determination': self.coeff_of_determination})
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
            print('Model successfully loaded from', load_path)
        except:
            print('Model could not be loaded from', self.model_dir)
            raise

        try:
            # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
            self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                               loss='mean_squared_error',
                               metrics=['mean_absolute_error', 'mean_squared_error', self.coeff_of_determination])
            print('Compilation successful.')
        except:
            print('Could not compile tf model.')
            raise

        self.load_normalizing_parameters()
        # self.load_model_parameters() #TODO activate this line as soon as new model is trained!

    def load_checkpoint(self, checkpoint_name='best'):
        load_path = os.path.join(self.model_dir, 'model_checkpoints')
        if checkpoint_name == 'latest':
            checkpoint = tf.train.latest_checkpoint(load_path)
        else:
            best_path = os.path.join(load_path, 'best')
            checkpoint = os.path.join(best_path, os.listdir(best_path)[0])
            print(checkpoint)

        try:
            self.model.load_weights(checkpoint)
            print('Checkpoint loaded successfully.')
        except:
            print('Could not load checkpoint.')
            raise

    def load_normalizing_parameters(self):
        try:
            load_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
            norm_params = pd.DataFrame().from_csv(load_path)
            self.means = norm_params['mean'].values
            self.stds = norm_params['standard deviation'].values
        except:
            print('Could not load normalization parameters from', load_path)
            raise
        return norm_params

    def load_model_parameters(self):
        load_path = os.path.join(self.model_dir, 'training_parameters.csv')
        model_params = pd.DataFrame().from_csv(load_path)
        values = model_params['values']
        self.input_sequence_length = values['input_sequence_length']
        self.time_step = values['time_step']
        return self.input_sequence_length, self.time_step


    def build_new_model(self, input_shape, layers=2, nodes_per_layer=32, activation_function='relu', regularization=0.01):
        print('input_shape',input_shape)
        inputs = tf.keras.Input(shape=(input_shape[0],input_shape[1],))

        if layers == 1:
            h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                     bias_regularizer=tf.keras.regularizers.l2(regularization))(inputs)
        elif layers > 2:
            h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                     bias_regularizer=tf.keras.regularizers.l2(regularization),
                                     return_sequences=True)(inputs)
            for layer in range(layers - 2):
                h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                         kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                         bias_regularizer=tf.keras.regularizers.l2(regularization),
                                         return_sequences=True)(h)
            h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                     bias_regularizer=tf.keras.regularizers.l2(regularization))(h)
        else:
            h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                     bias_regularizer=tf.keras.regularizers.l2(regularization),
                                     return_sequences=True)(inputs)
            h = tf.keras.layers.LSTM(nodes_per_layer, activation=activation_function,
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                     bias_regularizer=tf.keras.regularizers.l2(regularization))(h)

        predictions = tf.keras.layers.Dense(3, activation=None,
                                            kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                            bias_regularizer=tf.keras.regularizers.l2(regularization))(h)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate, self.learning_rate_decay),
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate, self.learning_rate_decay),
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error', self.coeff_of_determination])

        self.save_model_parameters(layers, nodes_per_layer, activation_function, regularization)

    def train_model(self, features, labels):
        # while not self.new_model:
        #     answer = input('There may be an existing model.\nWould you like to overwrite any existing models? (y/n)')
        #     if answer in ('y', 'Y', 'Yes', 'yes'):
        #         break
        #     elif answer in ('n', 'N', 'No', 'no'):
        #         print('Abort: Script will be terminated.')
        #         sys.exit()
        #     else:
        #         continue

        if not self.new_model:
            print(f'Model \"{self.model_name}\" already existing and trained.\nWill skip this model...')
            return

        if self.shuffle:
            features, labels = shuffle_list(features, labels, random_seed=self.random_seed)
        self.get_normalizing_parameters(features)
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

        # Callback: Tensorboard
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.root_folder + f'/logs/{self.model_name}')

        self.history = self.model.fit(normalized_features, labels, batch_size=self.batch_size, epochs=self.epochs,
                                      callbacks=[stop_early, save_checkpoints, save_best_checkpoint,
                                                 PrintState(self.model_name), tensor_board],
                                      validation_split=0.2, verbose=1)

        # Save entire model to a HDF5 file
        self.model.save(os.path.join(self.model_dir, 'my_model.h5'))
        # print('Training terminated\nModel saved successfully')

    def get_weights(self, layer='all'):
        W1 = []
        W2 = []
        B = []
        if layer == 'all':
            for i, l in enumerate(self.model.layers):
                if i == len(self.model.layers) - 1:
                    # print('Output Layer\n Weights {weights}\n Biases {biases}'.format(weights=l.get_weights()[0],
                    #                                                                   biases=l.get_weights()[1]))
                    W1.append(l.get_weights()[0])
                    W2.append(np.array([]))
                    B.append(l.get_weights()[1])
                elif i > 0:
                    # print('Layer {:5.0f}\n Weights {weights}\n Biases {biases}'.format(i, weights=l.get_weights()[0],
                    #                                                                    biases=l.get_weights()[1]))
                    W1.append(l.get_weights()[0])
                    W2.append(l.get_weights()[1])
                    B.append(l.get_weights()[2])
        elif isinstance(layer, int):
            l = self.model.layers[layer]
            # print('Layer {:5.0f}\n Weights {weights}\n Biases {biases}'.format(layer, weights=l.get_weights()[0],
            #                                                                    biases=l.get_weights()[1]))
            if len(l.get_weights()) > 2:
                W1.append(l.get_weights()[0])
                W2.append(l.get_weights()[1])
                B.append(l.get_weights()[2])
            else:
                W1.append(l.get_weights()[0])
                W2.append(np.array([]))
                B.append(l.get_weights()[1])
        else:
            print('layer argument must be an integer or \"all\"')
            raise ValueError
        return W1, W2, B

    def save_training_parameters(self):
        data = np.array([self.epochs, self.learning_rate, self.batch_size, self.shuffle, self.random_seed,
                         self.model_name, self.input_sequence_length, self.time_step]).transpose()
        training_parameters = pd.DataFrame(data=data,
                                           columns=['values'],
                                           index=['epochs', 'learning_rate', 'batch_size', 'shuffle', 'random_seed',
                                                  'model_name', 'input_sequence_length', 'time_step'])

        train_params_save_path = os.path.join(self.model_dir, 'training_parameters.csv')
        training_parameters.to_csv(train_params_save_path)

        data = np.array([self.means, self.stds]).transpose()
        normalizing_parameters = pd.DataFrame(data=data,
                                              columns=['mean', 'standard deviation'],
                                              index=['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
                                                     'pose vtheta [rad*s^-1]',
                                                     'steer position cal [n.a.]', 'brake position effective [m]',
                                                     'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]'])
        norm_params_save_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
        normalizing_parameters.to_csv(norm_params_save_path)

    def save_model_parameters(self, layers, nodes_per_layer, activation_function, regularization):
        data = np.array([layers, nodes_per_layer, activation_function, regularization]).transpose()
        model_parameters = pd.DataFrame(data=data,
                                        columns=['values'],
                                        index=['layers', 'nodes_per_layer', 'activation_function',
                                               'l2 regularization', ])

        model_params_save_path = os.path.join(self.model_dir, 'model_parameters.csv')
        model_parameters.to_csv(model_params_save_path)

    def save_training_history(self):
        save_path = os.path.join(self.model_dir, 'training_history')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.history is not None:
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch
            hist_save_path = os.path.join(save_path, 'history.csv')
            hist.to_csv(hist_save_path)

            # plt.figure('Mean Squared Error (Loss)')
            # plt.plot(hist[['epoch']].values, hist[['loss', 'val_loss']].values)
            # plt.legend(['Training Loss', 'Validation Loss'])
            # plt.savefig(os.path.join(save_path, 'loss_mean_squ_err.pdf'))
            #
            # plt.figure('Mean Absolute Error')
            # plt.plot(hist[['epoch']].values, hist[['mean_absolute_error', 'val_mean_absolute_error']].values)
            # plt.legend(['Training Error', 'Validation Error'])
            # plt.savefig(os.path.join(save_path, 'mean_abs_err.pdf'))
            #
            # plt.figure('Coefficient of Determination R^2')
            # plt.plot(hist[['epoch']].values, hist[['coeff_of_determination', 'val_coeff_of_determination']].values)
            # plt.legend(['Training', 'Validation'])
            # plt.savefig(os.path.join(save_path, 'R_squared.pdf'))
            #
            # plt.show()

        else:
            print('No training history found.')
            return 0

    def predict(self, input):
        input_normalized = self.normalize_data(input)
        tf.keras.backend.set_session(self.sess)
        with self.graph.as_default():
            result = self.model.predict(x=input_normalized, verbose=0)
        return result

    def save_model_performance(self, features, labels):
        if self.model_name != 'test' and self.history is not None:
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch
            best_index = np.argmin(hist['val_loss'].values)
            best = hist.iloc[best_index, :]
            best = pd.DataFrame(best).transpose()

            self.load_checkpoint()
            features_normalized = self.normalize_data(features)
            test_errors = self.model.evaluate(features_normalized, labels, verbose=0)

            best.insert(0, 'test_coeff_of_determination', test_errors[3])
            best.insert(0, 'test_mean_squared_error', test_errors[2])
            best.insert(0, 'test_mean_absolute_error', test_errors[1])
            best.insert(0, 'test_loss', test_errors[0])

            best.insert(0, 'model_name', self.model_name)
            best = best.round(3)
            with open(os.path.join(self.root_folder, 'model_performances.csv'), 'a') as f:
                best.to_csv(f, header=False, index=False)

    def test_model(self, features, labels):
        features_normalized = self.normalize_data(features)
        return self.model.evaluate(features_normalized, labels)

    def show_model_summary(self):
        return self.model.summary()

    def get_normalizing_parameters(self, training_features):
        training_features = training_features.reshape((1,-1,7))
        self.stds = np.std(training_features[0,:,:], axis=0)
        self.means = np.mean(training_features[0, :, :], axis=0)


    def normalize_data(self, features):
        centered = np.subtract(features, self.means)
        return np.divide(centered, self.stds)

    def coeff_of_determination(self, labels, predictions):
        total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labels, predictions)))
        r_squared = tf.subtract(1.0, tf.math.divide(unexplained_error, total_error))
        return r_squared


class PrintState(tf.keras.callbacks.Callback):
    def __init__(self, name):
        self.t0 = time.time()
        self.name = name

    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
            # print(logs)
            # print(type(logs))
            print(
                '{} Time: {:5.1f} s   Epoch: {:5.0f}    Training Loss: {:10.2f}    Validation Loss: {:10.2f}'.format(
                    self.name, time.time() - self.t0, epoch, logs['loss'], logs['val_loss']))
        # add callbacks=[PrintDot()] to model.fit(callbacks=[.....])