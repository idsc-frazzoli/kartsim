#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
# from parameter_optimization.model.marcsmodel_paramopt_scipy import marc_vehiclemodel
# from parameter_optimization.model.marcsmodel_paramopt_vectorized import marc_vehiclemodel as marc_vehiclemodel_vec
from parameter_optimization.model.marcsmodel_paramopt_vectorized_correct import marc_vehiclemodel as marc_vehiclemodel_vec_jcorr
from data_visualization.data_io import getPKL
import numpy as np
np.set_printoptions(precision=4)
import time
from scipy.optimize import least_squares, leastsq

def main():
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'

    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_no_of_samples = len(dataframe)
    print('Training on dataset with', total_no_of_samples, 'samples.')

    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].values

    # Initialize parameters
    # W = [1,1,1,1]
    W0 = np.array([9.4,10.4,0.3])

    np.random.shuffle(dataset)

    X = dataset

    def objective_function(W):
        # t0 = time.time()
        pred,_ = marc_vehiclemodel_vec_jcorr(X[:,:6], W)
        loss = 0.5 * np.sum(np.square(pred - X[:,6:]), axis=1)
        print('error',np.mean(loss),'weights',W)
        # print('time', time.time()-t0)
        return loss

    # res = leastsq(objective_function, W0, ftol=0.0001)
    # print(res[0])

    # res = least_squares(objective_function, W0, bounds=([0, 0, 1.05, -np.inf], [50, 50, 1.1, np.inf]))
    res = least_squares(objective_function, W0)
    print('res.x',res.x)

    print('Optimization finished.')

if __name__ == '__main__':
    main()