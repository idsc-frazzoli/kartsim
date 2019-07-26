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

        # Load parameters
        self.sequence_length, self.time_step = self.lstm.load_model_parameters()
        self.time_for_next_sequence_element = self.time_step
        self.disturbance = [0, 0, 0]

        self.data_sequence = deque(maxlen=self.sequence_length)

        # Load nominal vehicle model
        self.mpc = DynamicVehicleMPC()

    def get_name(self):
        return self.name

    def reinitialize_variables(self):
        self.time_for_next_sequence_element = self.time_step
        self.disturbance = [0, 0, 0]
        self.data_sequence = deque(maxlen=self.sequence_length)

    def get_accelerations(self, t, initial_velocities=[0, 0, 0], system_inputs=[0, 0, 0, 0]):
        if isinstance(initial_velocities, list):
            if len(self.data_sequence) < self.sequence_length:
                input = np.hstack((initial_velocities, system_inputs))
                self.data_sequence.append(input)
                if len(self.data_sequence) == self.sequence_length:
                    self.disturbance = self.lstm.predict(np.array([self.data_sequence]))
            elif t >= self.time_for_next_sequence_element:
                self.time_for_next_sequence_element += self.time_step
                input = np.hstack((initial_velocities, system_inputs))
                self.data_sequence.append(input)
                if len(self.data_sequence) == self.sequence_length:
                    self.disturbance = self.lstm.predict(np.array([self.data_sequence]))
            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            result = np.array(accelerations).transpose() + self.disturbance
            return result[0]
        elif np.array(initial_velocities).shape[0] >= self.sequence_length:
            self.disturbance = np.zeros((self.sequence_length-1,3))
            sequence = deque(maxlen=self.sequence_length)
            input = np.hstack((initial_velocities, system_inputs))
            sequential_data = []
            for row in input:
                sequence.append(row)
                if len(sequence) == self.sequence_length:
                    sequential_data.append(np.array(sequence))
            self.disturbance = np.vstack((self.disturbance,self.lstm.predict(sequential_data)))
            accelerations = self.mpc.get_accelerations(initial_velocities, system_inputs)
            result = np.array(accelerations).transpose() + self.disturbance
            return result
