#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10.07.19 10:25

@author: mvb
"""

import tensorflow as tf
import numpy as np
from learned_model.ml_models.rnn_keras import LongShortTermMemoryNetwork
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from collections import deque


class HybridLSTMModel:

    def __init__(self, model_name='FirstTry'):
        self.name = "hybrid_lstm"

        # Load the NN
        self.lstm = LongShortTermMemoryNetwork(model_name=model_name, predict_only=True)
        self.lstm.load_model()
        self.lstm.load_checkpoint('best')
        # self.weights, self.biases = self.lstm.get_weights()

        # Load parameters for normalizing inputs
        norm_params = self.lstm.load_normalizing_parameters()
        self.means = norm_params['mean'].values
        self.stds = norm_params['standard deviation'].values
        self.sequence_length = 5
        self.time_step = 0.1
        self.time_for_next_sequence_element = 0
        self.disturbance = [0, 0, 0]
        # self.sequence_length, self.time_step = self.lstm.load_model_parameters() #TODO activate this line as soon as new model is trained!

        self.data_sequence = deque(maxlen=self.sequence_length)

        # Load nominal vehicle model
        self.mpc = DynamicVehicleMPC()

    def get_name(self):
        return self.name

    def normalize_input(self, input):
        return (input - self.means) / self.stds

    def get_accelerations(self, time, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if time == 0.0:
            # print('1')
            self.time_for_next_sequence_element = self.time_step
            input = np.hstack((initial_velocities, system_inputs))
            self.data_sequence.append(input)
        elif time < self.time_step:
            # print('2')
            if len(self.data_sequence) < self.sequence_length and self.disturbance == [0, 0, 0]:
                input = np.hstack((initial_velocities, system_inputs))
                self.data_sequence.append(input)
            else:
                self.disturbance = self.lstm.predict(np.array([self.data_sequence]))
        elif time >= self.time_for_next_sequence_element:
            # print('3')
            self.time_for_next_sequence_element += self.time_step
            input = np.hstack((initial_velocities, system_inputs))
            self.data_sequence.append(input)
            if len(self.data_sequence) == self.sequence_length:
                self.disturbance = self.lstm.predict(np.array([self.data_sequence]))

        accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
        # print(f'disturbance {self.disturbance}  velocity {initial_velocities}')
        # print('accelerations', np.array(accelerations).transpose())
        # print(time)
        result = np.array(accelerations).transpose() + self.disturbance

        return result[0]
