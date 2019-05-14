#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
from parameter_optimization.model.marcsmodel_paramopt_own import marc_vehiclemodel, Jacobian_marc_vehiclemodel
from dataanalysisV2.dataIO import getPKL
import numpy as np
np.set_printoptions(precision=4)
import time

def main():
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/33333333333333/20190404T133714_01_sampledlogdata.pkl'
    batch_size = 100
    learning_rate = 0.01
    epochs = 10

    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_no_of_samples = len(dataframe)
    no_of_batches = int(total_no_of_samples / batch_size)
    print('Training on dataset with', total_no_of_samples, 'samples.')

    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].values

    # Initialize parameters
    # W = [1,1,1,1]
    W = [9.4,10.4,1.08,0.3]

    # for epoch in range(epochs):
    #     # Shuffle dataset
    #     # np.random.seed(4)
    #     np.random.shuffle(dataset)
    #
    #     for batch_no in range(no_of_batches):
    #         if batch_no != no_of_batches:
    #             batch = dataset[batch_no * batch_size : (batch_no+1) * batch_size,:]
    #         else:
    #             batch = dataset[batch_no * batch_size: , :]
    #         predictions = []
    #         for X in batch:
    #             predictions.append(marc_vehiclemodel(X[:6], W))
    #
    #         loss, error = loss_function(predictions, batch[:,6:])
    #
    #         J = Jacobian_marc_vehiclemodel(X[:6], W)
    #
    #         W = W - learning_rate * np.matmul(J.transpose(), error)
    #
    #         if batch_no % 10 == 0:
    #             print('epoch:', epoch, 'loss:', error, 'weights:', W)
    # print('Optimization finished.')
    # print('epoch:', epoch, 'loss:', loss, 'weights:', W)


def loss_function(pred, target):
    error = np.mean(np.subtract(pred, target),axis=0)
    loss = np.sum(np.mean(np.square(np.subtract(pred, target)),axis=0))
    return loss, np.array(error)

if __name__ == '__main__':
    main()