#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
# from parameter_optimization.model.marcsmodel_paramopt_scipy import marc_vehiclemodel
from parameter_optimization.model.marcsmodel_paramopt_vectorized import marc_vehiclemodel
from dataanalysisV2.dataIO import getPKL
import numpy as np
np.set_printoptions(precision=4)
import time
from scipy.optimize import least_squares, leastsq

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
    W0 = np.array([9.4,10.4,1.08,0.3])

    np.random.shuffle(dataset)

    X = dataset

    def objective_function(W):
        # print(marc_vehiclemodel(X[:,:6], W).shape)
        # print(X[:,6:].shape)
        # data = np.reshape(X[:,6:], (1,0))
        pred = marc_vehiclemodel(X[:,:6], W)
        square = np.square(pred - X[:,6:])
        loss = 0.5 * np.sum(square, axis=1)
        error = loss
        # print('pred',pred[0])
        # print('pred',pred.shape)
        # print('target', X[:,6:][0])
        # print('target', X[:,6:].shape)
        # print('square', square[0])
        # print('sum',sum)
        print('error',np.mean(error),'weights',W)
        return loss

    # res = leastsq(objective_function, W0, ftol=0.0001)
    # print(res[0])

    # res = least_squares(objective_function, W0, bounds=([0, 0, 1.05, -np.inf], [50, 50, 1.1, np.inf]))
    res = least_squares(objective_function, W0)
    print('res.x',res.x)

    print('Optimization finished.')

if __name__ == '__main__':
    main()