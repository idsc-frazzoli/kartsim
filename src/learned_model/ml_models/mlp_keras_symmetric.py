#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 13.06.19 13:21

@author: mvb
"""
import numpy as np
import pandas as pd
import os
import tensorflow as tf

import config
from gokart_data_preprocessing.shuffle import shuffle_dataframe
from data_visualization.data_io import create_folder_with_time, getDirectories
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import time


class MultiLayerPerceptronSymmetric():
    def __init__(self, epochs=20, learning_rate=1e-4, batch_size=100, shuffle=True, random_seed=None, model_name='test',
                 predict_only=False):
        self.root_folder = os.path.join(config.directories['root'], 'Models/trained_mlp_models')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.model_name = model_name
        self.model = None
        self.history = None
        self.new_model = False
        self.train_stats = pd.DataFrame()

        if predict_only:
            tf.keras.backend.set_learning_phase(0)

        self.model_dir = None
        folder_names = getDirectories(self.root_folder)
        for name in folder_names:
            if name.endswith(model_name):
                print('Model name already exists!')
                self.model_dir = os.path.join(self.root_folder, name)

        if self.model_dir is None:
            # print('New model name! Creating new folder...')
            self.model_dir = create_folder_with_time(self.root_folder, model_name)
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints'))
            os.mkdir(os.path.join(self.model_dir, 'model_checkpoints', 'best'))
            self.new_model = True

    def load_model(self):
        try:
            load_path = os.path.join(self.model_dir, 'my_model.h5')
            self.model = tf.keras.models.load_model(load_path, custom_objects={
                'coeff_of_determination': self.coeff_of_determination, 'swap_rimo': self.swap_rimo})
            # print('Model successfully loaded from', load_path)
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

        try:
            load_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
            norm_params = pd.DataFrame().from_csv(load_path)
            self.train_stats = norm_params
            self.train_stats.columns = ['mean', 'std']
        except:
            print('Could not load normalization parameters from', load_path)
            raise

    def get_normalizing_parameters(self):
        return self.train_stats['mean'].values, self.train_stats['std'].values

    def load_checkpoint(self, checkpoint_name='best'):
        load_path = os.path.join(self.model_dir, 'model_checkpoints')

        if checkpoint_name == 'latest':
            checkpoint = tf.train.latest_checkpoint(load_path)
        else:
            best_path = os.path.join(load_path, 'best')
            checkpoint = os.path.join(best_path, os.listdir(best_path)[0])
        try:
            self.model.load_weights(checkpoint)
            # print('Checkpoint loaded successfully.')
        except:
            print('Could not load checkpoint.')
            raise

    def load_normalizing_parameters(self):
        try:
            load_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
            norm_params = pd.DataFrame().from_csv(load_path)
            self.train_stats = norm_params
            self.train_stats.columns = ['mean', 'std']
            return pd.DataFrame().from_csv(load_path)
        except:
            print('Could not load normalization parameters from', load_path)
            raise

    def swap_rimo(self, tensor):
        U = tf.keras.backend.constant([[1., 0., 0., 0., 0., 0., 0.], [0., -1., 0., 0., 0., 0., 0.], [0., 0., -1., 0., 0., 0., 0.],
                         [0., 0., 0., -1., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 0., 1., 0.]])
        return tf.keras.backend.dot(tensor, U)

    def build_new_model(self, input_dim, output_dim=3, layers=2, nodes_per_layer=32, activation_function=None, regularization=0.01):
        inputs = tf.keras.Input(shape=(input_dim,))
        # inputs_sym = tf.keras.layers.Lambda(lambda inputs: inputs * np.array([1.,-1.,-1.,-1.,1.,1.,1.]))(inputs)
        # rimo_l = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., 0., 0., 0., 0., 1., 0.]])])
        # rimo_r = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., 0., 0., 0., 0., 0., 1.]], dtype=tf.float32)])
        # vx = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[1., 0., 0., 0., 0., 0., 0.]], dtype=tf.float32)])
        # vy = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., -1., 0., 0., 0., 0., 0.]], dtype=tf.float32)])
        # vtheta = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., 0., -1., 0., 0., 0., 0.]], dtype=tf.float32)])
        # steer = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., 0., 0., -1., 0., 0., 0.]], dtype=tf.float32)])
        # brake = tf.keras.layers.Dot(axes=-1)([inputs_sym, tf.constant([[0., 0., 0., 0., 1., 0., 0.]], dtype=tf.float32)])
        # # input_wo_rimo = tf.keras.layers.Dot(axes=-1)([inputs, tf.constant([[1., 1., 1., 1., 1., 0., 0.]], dtype=tf.float32)])
        # inputs_sym = tf.keras.layers.Concatenate(axis=-1)([vx, vy, vtheta, steer, brake, rimo_r, rimo_l])

        # const1 = tf.constant([
        #     [1., 0., 0., 0., 0., 0., 0.],
        #     [0., 1., 0., 0., 0., 0., 0.],
        #     [0., 0., 1., 0., 0., 0., 0.],
        #     [0., 0., 0., 1., 0., 0., 0.],
        #     [0., 0., 0., 0., 1., 0., 0.],
        #     [0., 0., 0., 0., 0., 0., 1.],
        #     [0., 0., 0., 0., 0., 1., 0.]])

        # swap_rimo = self.swap_rimo

        # rimo_l = tf.keras.layers.Dot(axes=-1)([L, np.array([[0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,1.,0.]]).T])
        inputs_sym = tf.keras.layers.Lambda(self.swap_rimo)(inputs)

        # odd symmetry
        if layers > 0:
            dense1 = tf.keras.layers.Dense(nodes_per_layer, activation=activation_function,
                                           kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                           bias_regularizer=tf.keras.regularizers.l2(regularization),)
            if layers == 1:
                h_pos = dense1(inputs)
                h_neg = dense1(inputs_sym)
            else:
                # dense2 = tf.keras.layers.Dense(nodes_per_layer, activation=activation_function,
                #                                kernel_regularizer=tf.keras.regularizers.l2(regularization),
                #                                bias_regularizer=tf.keras.regularizers.l2(regularization))
                h_pos = dense1(inputs)
                h_neg = dense1(inputs_sym)
                for layer in range(layers - 1):
                    # h_pos = dense2(h_pos)
                    # h_neg = dense2(h_neg)
                    dense = tf.keras.layers.Dense(nodes_per_layer, activation=activation_function,
                                                  kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                                  bias_regularizer=tf.keras.regularizers.l2(regularization))
                    h_pos = dense(h_pos)
                    h_neg = dense(h_neg)

            list = [h_pos, h_neg]
            #for even function
            h_even = tf.keras.layers.Average()(list)
            # for odd function
            h_odd = tf.keras.layers.Subtract()(list)
            h_odd = tf.keras.layers.Lambda(lambda inputs: inputs / 2.0)(h_odd)

            predictions_even = tf.keras.layers.Dense(1, activation=None,
                                                kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                                bias_regularizer=tf.keras.regularizers.l2(regularization),)(h_even)
            predictions_odd = tf.keras.layers.Dense(2, activation=None,
                                                     kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                                     bias_regularizer=tf.keras.regularizers.l2(regularization),
                                                     use_bias=False)(h_odd)
            predictions = tf.keras.layers.Concatenate(axis=-1)([predictions_even, predictions_odd])
        else:
            dense1 = tf.keras.layers.Dense(1, activation=activation_function,
                                           kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                           bias_regularizer=tf.keras.regularizers.l2(regularization), )
            dense2 = tf.keras.layers.Dense(2, activation=activation_function,
                                           kernel_regularizer=tf.keras.regularizers.l2(regularization),
                                           bias_regularizer=tf.keras.regularizers.l2(regularization), )
            h_pos_1 = dense1(inputs)
            h_neg_1 = dense1(inputs_sym)
            h_pos_2 = dense2(inputs)
            h_neg_2 = dense2(inputs_sym)

            # for even function
            predictions_even = tf.keras.layers.Average()([h_pos_1, h_neg_1])
            # for odd function
            predictions_odd = tf.keras.layers.Subtract()([h_pos_2, h_neg_2])
            predictions_odd = tf.keras.layers.Lambda(lambda inputs: inputs / 2.0)(predictions_odd)

            predictions = tf.keras.layers.Concatenate(axis=-1)([predictions_even, predictions_odd])


        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
        # self.model.compile(optimizer=tf.keras.optimizers.Adadelta(),
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
            features, labels = shuffle_dataframe(features, labels, random_seed=self.random_seed)

        self.get_train_stats(features)
        # normalized_features, normalized_labels = self.normalize_data(features, labels)
        normalized_features = self.normalize_data(features)
        # if 'sym' in self.model_name:
        #     normalized_features_sym = self.mirror_state_space(normalized_features)

        self.save_training_parameters()

        self.save_path_checkpoints = os.path.join(self.model_dir, 'model_checkpoints', 'mpl-{epoch:04d}.ckpt')
        self.save_path_best_checkpoint = os.path.join(self.model_dir, 'model_checkpoints', 'best', 'mpl-best.ckpt')

        # Callback: Save model checkpoints regularly
        save_checkpoints = tf.keras.callbacks.ModelCheckpoint(self.save_path_checkpoints, verbose=0, period=5)

        # Callback: Save best model checkpoint
        save_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.save_path_best_checkpoint, verbose=0,
                                                                  save_best_only=True, monitor='val_loss', mode='min')

        # Callback: Stop training if validation loss does not improve anymore
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300)

        # Callback: Tensorboard
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.root_folder + f'/logs/{self.model_name}')

        self.history = self.model.fit(normalized_features, labels, batch_size=self.batch_size, epochs=self.epochs,
                                      callbacks=[stop_early, save_checkpoints, save_best_checkpoint,
                                                 PrintState(self.model_name), tensor_board],
                                      validation_split=0.2, verbose=0)

        # Save entire model to a HDF5 file
        self.model.save(os.path.join(self.model_dir, 'my_model.h5'))
        # print('Training terminated\nModel saved successfully')

    def get_weights(self, layer='all'):
        W = []
        B = []
        if layer == 'all':
            for i, l in enumerate(self.model.layers):
                weights = l.get_weights()
                if len(weights) > 0:
                    if len(weights) > 1:
                        W.append(l.get_weights()[0])
                        B.append(l.get_weights()[1])
                    else:
                        W.append(l.get_weights()[0])
                        B.append(np.array([]))
        elif isinstance(layer, int):
            l = self.model.layers[layer]
            # print('Layer {:5.0f}\n Weights {weights}\n Biases {biases}'.format(layer, weights=l.get_weights()[0],
            #                                                                    biases=l.get_weights()[1]))
            W.append(l.get_weights()[0])
            B.append(l.get_weights()[1])
        else:
            print('layer argument must be an integer or \"all\"')
            raise ValueError
        return W, B

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
        index_list = []
        for index in self.train_stats.index:
            index_list.append(index)
        normalizing_parameters = pd.DataFrame(data=data,
                                              columns=['mean', 'standard deviation'],
                                              index=index_list)

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

    def predict(self, features):
        features_normalized = self.normalize_data(features)
        # print(features)
        # print(features_normalized)
        # if 'sym' in self.model_name:
        #     features_normalized_symmetric = self.mirror_state_space(features_normalized)
        #     print(features_normalized, features_normalized_symmetric)
        # result = self.model.predict(x=[features_normalized, features_normalized_symmetric], verbose=0)
        result = self.model.predict(x=features_normalized, verbose=0)
        # result[:, 1:3] = result[:, 1:3] * np.column_stack((label_mirror, label_mirror))
        return result

    def save_model_performance(self, test_features, test_labels):
        if self.model_name != 'test' and self.history is not None:
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch
            best_index = np.argmin(hist['val_loss'].values)
            best = hist.iloc[best_index, :]
            best = pd.DataFrame(best).transpose()

            self.load_checkpoint()
            test_errors = self.test_model(test_features, test_labels)

            best.insert(0, 'test_coeff_of_determination', test_errors[3])
            best.insert(0, 'test_mean_squared_error', test_errors[2])
            best.insert(0, 'test_mean_absolute_error', test_errors[1])
            best.insert(0, 'test_loss', test_errors[0])

            best.insert(0, 'model_name', self.model_name)
            best = best.round(3)
            with open(os.path.join(self.root_folder, 'model_performances.csv'), 'a') as f:
                best.to_csv(f, header=False, index=False)

    def test_model(self, test_features, test_labels):
        features_normalized = self.normalize_data(test_features)
        # if 'sym' in self.model_name:
        #     features_normalized_symmetric = self.mirror_state_space(test_features)
        # features_normalized, labels_normalized = self.normalize_data(test_features, test_labels)
        return self.model.evaluate(features_normalized, test_labels, verbose=0)

    def show_model_summary(self):
        return self.model.summary()

    def get_train_stats(self, training_features):
        # feature_labels = training_features.join(training_labels)
        self.train_stats = training_features.describe()
        self.train_stats = self.train_stats.transpose()
        self.train_stats.loc['motor torque cmd left [A_rms]':'motor torque cmd right [A_rms]','std'] = \
            np.mean(self.train_stats.loc['motor torque cmd left [A_rms]':'motor torque cmd right [A_rms]','std'].values)

    def normalize_data(self, features):
        if isinstance(features, pd.DataFrame):
            features.iloc[:,np.array([0,4])] = features.iloc[:,np.array([0,4])] - self.train_stats['mean'][np.array([0,4])]
            return features / self.train_stats['std'].values
            # return features / self.train_stats['std'][:6], (labels - self.train_stats['mean'][6:]) / self.train_stats['std'][6:]
            # return (features - self.train_stats['mean']) / self.train_stats['std']
        else:
            features[:, 0] = features[:, 0] - self.train_stats['mean'][0]
            features[:, 4] = features[:, 4] - self.train_stats['mean'][4]
            return features / self.train_stats['std'].values
            # return (features - self.train_stats['mean'].values) / self.train_stats['std'].values

    def mirror_state_space(self, raw_features):
        if isinstance(raw_features, pd.DataFrame):
            features = np.copy(raw_features.values)
        else:
            features = np.copy(raw_features)

        features = features * np.array([1,-1,-1,-1,1,-1])

        if isinstance(raw_features, pd.DataFrame):
            raw_features.loc[:, :] = features
            return raw_features
        else:
            return features

    def mirror_states(self, features):
        pass

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
        if epoch % 20 == 0:
            # print(logs)
            # print(type(logs))
            print(
                '{} Time: {:5.1f} s   Epoch: {:4.0f}    Training Loss: {:5.4f}    Validation Loss: {:5.4f}'.format(
                    self.name, time.time() - self.t0, epoch, logs['loss'], logs['val_loss']))
        # add callbacks=[PrintDot()] to model.fit(callbacks=[.....])
