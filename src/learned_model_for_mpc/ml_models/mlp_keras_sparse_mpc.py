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
from tensorflow_model_optimization.python.core.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, \
    PruningSummaries

deprecation._PRINT_DEPRECATION_WARNINGS = False
import time


class MultiLayerPerceptronMPCSparse():
    def __init__(self, epochs=1000, learning_rate=1e-4, batch_size=100, shuffle=True, random_seed=None,
                 model_name='test',
                 predict_only=False):
        self.root_folder = os.path.join(config.directories['root'], 'Models/trained_mlp_no_models')
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
            # load_path = os.path.join(self.model_dir, 'my_model.h5')
            load_path = os.path.join(self.model_dir, 'my_pruned_model.h5')
            with sparsity.prune.prune_scope():
                self.pruned_model = tf.keras.models.load_model(load_path, custom_objects={
                    'coeff_of_determination': self.coeff_of_determination})
            # print('Model successfully loaded from', load_path)
        except:
            print('Model could not be loaded from', self.model_dir)
            raise

        try:
            # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
            self.pruned_model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
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

    def load_checkpoint(self, checkpoint='best'):
        load_path = os.path.join(self.model_dir, 'model_checkpoints')
        with sparsity.prune.prune_scope():
            if isinstance(checkpoint, str):
                if checkpoint == 'latest':
                    file_list = os.listdir(load_path)
                    file_list.sort()
                    print(file_list[-1])
                    checkpoint = os.path.join(load_path, file_list[-1])
                    # checkpoint = tf.train.latest_checkpoint(load_path)
                else:
                    best_path = os.path.join(load_path, 'best')
                    checkpoint = os.path.join(best_path, os.listdir(best_path)[0])
            else:
                file_list = os.listdir(load_path)
                file_list.sort()
                print(file_list[checkpoint])
                checkpoint = os.path.join(load_path, file_list[checkpoint])
            try:
                pruning_params = {
                    'pruning_schedule': PolynomialDecay(initial_sparsity=0.50,
                                                        final_sparsity=0.99,
                                                        begin_step=5000,
                                                        end_step=100000,
                                                        frequency=100)
                }
                # self.pruned_model = sparsity.prune.prune_low_magnitude(self.pruned_model, **pruning_params)
                self.pruned_model.load_weights(checkpoint)
                # print('Checkpoint loaded successfully.')
            except:
                print('Could not load checkpoint.')
                raise

    def load_normalizing_parameters(self):
        load_path = os.path.join(self.model_dir, 'normalizing_parameters.csv')
        return pd.DataFrame().from_csv(load_path)

    def get_normalizing_parameters(self):
        return self.train_stats['mean'].values, self.train_stats['std'].values

    def build_new_model(self, input_dim, output_dim, layers=2, nodes_per_layer=32, activation_function=None,
                        regularization=0.01, end_step=0):
        # introduce pruning
        pruning_params = {
            'pruning_schedule': PolynomialDecay(initial_sparsity=0.0,
                                                final_sparsity=0.90,
                                                begin_step=end_step/5,
                                                end_step=end_step,
                                                frequency=end_step/500)
            # 'pruning_schedule': PolynomialDecay(initial_sparsity=0.2,
            #                                     final_sparsity=0.90,
            #                                     begin_step=end_step / 5,
            #                                     end_step=end_step,
            #                                     frequency=100)
        }

        inputs = tf.keras.Input(shape=(input_dim,))
        predictions = tf.keras.layers.Dense(output_dim, activation=None,
                                                kernel_regularizer=tf.keras.regularizers.l1(regularization),
                                                bias_regularizer=tf.keras.regularizers.l1(regularization))(inputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

        self.pruned_model = sparsity.prune.prune_low_magnitude(self.model, **pruning_params)

        # self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
        self.pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
        # self.model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error', self.coeff_of_determination])

        self.save_model_parameters(layers, nodes_per_layer, activation_function, regularization)

    def build_train_test_model(self, train_features, train_labels, test_features=None, test_labels=None, layers=2,
                               nodes_per_layer=32, activation_function=None, regularization=0.01):
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
            train_features, train_labels = shuffle_dataframe(train_features, train_labels, random_seed=self.random_seed)

        if 'sym' in self.model_name:
            train_features, train_labels = self.mirror_state_space(train_features, train_labels)
        # train_features = self.add_features(train_features)
        self.get_train_stats(train_features)
        normalized_features = self.normalize_data(train_features)

        # parameters for pruning
        num_train_samples = train_features.shape[0]
        end_step = np.ceil(1.0 * num_train_samples / self.batch_size).astype(np.int32) * self.epochs

        self.build_new_model(input_dim=normalized_features.shape[1], output_dim=train_labels.shape[1],
                             layers=layers, nodes_per_layer=nodes_per_layer,
                             activation_function=activation_function, regularization=regularization, end_step=end_step)

        self.save_training_parameters()

        self.save_path_checkpoints = os.path.join(self.model_dir, 'model_checkpoints', 'mpl-{epoch:04d}.ckpt')
        self.save_path_best_checkpoint = os.path.join(self.model_dir, 'model_checkpoints', 'best', 'mpl-best.ckpt')

        # Callback: Save model checkpoints regularly
        save_checkpoints = tf.keras.callbacks.ModelCheckpoint(self.save_path_checkpoints, verbose=0, period=5)

        # Callback: Save best model checkpoint
        save_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.save_path_best_checkpoint, verbose=0,
                                                                  save_best_only=True, monitor='val_loss', mode='min')

        # Callback: Stop training if validation loss does not improve anymore
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

        # Callback: Tensorboard
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.root_folder + f'/logs/{self.model_name}')

        # Update the pruning step
        pruning_update_step = UpdatePruningStep()
        # Add summaries to keep track of the sparsity in different layers during training
        pruning_summaries = PruningSummaries(log_dir=self.root_folder + f'/logs/{self.model_name}', profile_batch=0)

        self.history = self.pruned_model.fit(normalized_features, train_labels, batch_size=self.batch_size, epochs=self.epochs,
                                      callbacks=[stop_early, save_checkpoints, save_best_checkpoint,
                                                 PrintState(self.model_name), tensor_board,
                                                 pruning_update_step, pruning_summaries],
                                      validation_split=0.2, verbose=0)


        # Save entire model to a HDF5 file
        tf.keras.models.save_model(self.pruned_model, os.path.join(self.model_dir, 'my_pruned_model.h5'), include_optimizer=True)
        self.model = sparsity.prune.strip_pruning(self.pruned_model)
        self.model.save(os.path.join(self.model_dir, 'my_model.h5'))
        self.save_training_history()
        if test_features is not None and test_labels is not None:
            self.save_model_performance(test_features, test_labels)
        # print('Training terminated\nModel saved successfully')

    def get_weights(self, layer='all'):
        self.model = sparsity.prune.strip_pruning(self.pruned_model)
        W = []
        B = []
        if layer == 'all':
            for i, l in enumerate(self.model.layers):
                if i == len(self.model.layers) - 1:
                    # print('Output Layer\n Weights {weights}\n Biases {biases}'.format(weights=l.get_weights()[0],
                    #                                                                   biases=l.get_weights()[1]))
                    W.append(l.get_weights()[0])
                    B.append(l.get_weights()[1])
                elif i > 0:
                    # print('Layer {:5.0f}\n Weights {weights}\n Biases {biases}'.format(i, weights=l.get_weights()[0],
                    #                                                                    biases=l.get_weights()[1]))
                    W.append(l.get_weights()[0])
                    B.append(l.get_weights()[1])
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
                                               'l1 regularization', ])

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
        label_mirror = np.ones(len(features))
        if 'sym' in self.model_name:
            features, label_mirror = self.mirror_state_space(features)
        # features = self.add_features(features)
        features_normalized = self.normalize_data(features)
        result = self.model.predict(x=features_normalized, verbose=0)
        result[:, 1:3] = result[:, 1:3] * np.column_stack((label_mirror, label_mirror))
        return result

    def save_model_performance(self, test_features, test_labels):
        if self.model_name != 'test' and self.history is not None:
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch
            # best_index = np.argmin(hist['val_loss'].values)
            best = hist.iloc[-1, :]
            best = pd.DataFrame(best).transpose()

            # self.load_checkpoint()
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
        if 'sym' in self.model_name:
            test_features, test_labels = self.mirror_state_space(test_features, test_labels)
        # test_features = self.add_features(test_features)
        features_normalized = self.normalize_data(test_features)
        return self.pruned_model.evaluate(features_normalized, test_labels, verbose=0)

    def show_model_summary(self):
        return self.pruned_model.summary()

    def get_train_stats(self, training_features):
        self.train_stats = training_features.describe()
        self.train_stats = self.train_stats.transpose()

    def add_features(self, raw_features):
        if isinstance(raw_features, pd.DataFrame):
            features = raw_features.copy()
            features = features.iloc[:, 3:]
            return features

            # for feature in features:
            #     new_feature = feature + '^2'
            #     features.loc[:, new_feature] = np.square(features[feature].values)
            # features[features == 0] = 0.000001
            # for feature in features:
            #     new_feature = feature + '^-1'
            #     features.loc[:, new_feature] = np.divide(1, features[feature].values)
            # return features
        else:
            features = np.copy(raw_features)
            features = features[:, 3:]
            return features

            # for feature in features.T:
            #     features = np.hstack((features, np.array([np.square(feature)]).T))
            # features[features == 0] = 0.000001
            # for feature in features.T:
            #     features = np.hstack((features, np.array([np.divide(1, feature)]).T))
            # return features

    def normalize_data(self, features):
        if isinstance(features, pd.DataFrame):
            return (features - self.train_stats['mean']) / self.train_stats['std']
        else:
            return (features - self.train_stats['mean'].values) / self.train_stats['std'].values

    def mirror_state_space(self, raw_features, raw_labels=None):
        if isinstance(raw_features, pd.DataFrame):
            features = np.copy(raw_features.values)
            if raw_labels is not None:
                labels = np.copy(raw_labels.values)
        else:
            features = np.copy(raw_features)
            if raw_labels is not None:
                labels = np.copy(raw_labels)

        steer_less_zero = features[:, 3] < 0
        mirror_vel_steer = steer_less_zero * -2 + 1
        features[:, 1:4] = features[:, 1:4] * np.column_stack((mirror_vel_steer, mirror_vel_steer, mirror_vel_steer))
        # features[steer_less_zero, 5], features[steer_less_zero, 6] = features[steer_less_zero, 6], features[steer_less_zero, 5]
        features[:, -1] = features[:, -1] * mirror_vel_steer
        if raw_labels is not None:
            labels[:, 1:3] = labels[:, 1:3] * np.column_stack((mirror_vel_steer, mirror_vel_steer))
        if isinstance(raw_features, pd.DataFrame):
            raw_features.loc[:, :] = features
            if raw_labels is not None:
                raw_labels.loc[:, :] = labels
                return raw_features, raw_labels
            return raw_features
        else:
            if raw_labels is not None:
                return features, labels
            return features, mirror_vel_steer

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
                '{} Time: {:5.1f} s   Epoch: {:5.0f}    Training Loss: {:10.2f}    Validation Loss: {:10.2f}'.format(
                    self.name, time.time() - self.t0, epoch, logs['loss'], logs['val_loss']))
        # add callbacks=[PrintDot()] to model.fit(callbacks=[.....])
