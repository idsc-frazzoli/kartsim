#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 17.05.19 10:24

@author: mvb
"""

# from parameter_optimization.model.marcsmodel_paramopt_vectorized import marc_vehiclemodel as marc_vehiclemodel_vec
import parameter_optimization.integrate.timeIntegrators as integrators
# import simulator.integrate.timeIntegrators as integrators
from dataanalysisV2.dataIO import getPKL
import numpy as np
np.set_printoptions(precision=4)
import time
from scipy.optimize import least_squares

def main():
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'
    validation_horizon = 1.0
    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_no_of_samples = len(dataframe)
    print('Training on dataset with', total_no_of_samples, 'samples.')

    time_vals = dataframe['time [s]'].values
    dt = time_vals[1]-time_vals[0]
    interval = int(np.round(validation_horizon/dt))

    split_rows = np.where(time_vals==0)[0]

    dataset = dataframe[['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]',
                         'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                         'pose atheta [rad*s^-2]', 'MH BETA [rad]', 'MH AB [m*s^-2]', 'MH TV [rad*s^-2]']].values

    X0 = []
    sim_time = []
    U = []
    # target = []
    for start,stop in zip(split_rows,np.append(split_rows[1:],len(time_vals))):
        for i in range(int(np.floor((stop-start)/interval))):
            # print(' ',start + i * interval , start + (i + 1) * interval)
            chunk = dataset[start + i * interval : start + (i + 1) * interval]
            X0.append(chunk[0, :7])
            sim_time.append(validation_horizon)
            U.append(np.hstack((chunk[:,0:1],chunk[:,10:])))
            if i==0:
                target = chunk[:, 1:7]
            else:
                target = np.vstack((target,chunk[:, 1:7]))

        # print(' ',start + int(np.floor((stop-start)/interval)) * interval, stop)
        chunk = dataset[start + int(np.floor((stop-start)/interval)) * interval: stop]
        X0.append(chunk[0, :7])
        sim_time.append(chunk[-1,0] - chunk[0,0])
        U.append(np.hstack((chunk[:,0:1],chunk[:,10:])))
        target = np.vstack((target,chunk[:-1, 1:7]))

    # print('dataset',dataset[0])
    # print('X0',X0[0])
    # print('sim_time', sim_time[0])
    # print('U', U[0])
    # print('target', target[0])



    # Initialize parameters [D1,D2,Ic]
    W0 = [0.1,0.1,.1]
    # W0 = np.array([9.4, 10.4, 0.3])
    # t0 = time.time()
    # np.random.shuffle(dataset)

    def objective_function(W):
        XW = np.append(X0[0], W0)
        predictions = integrators.odeIntegratorIVP(XW, U[0].transpose(), sim_time[0], 0.01)[:-1, 1:-3]
        for i in range(1, len(X0)):
            XW = np.append(X0[i], W0)
            pred = integrators.odeIntegratorIVP(XW, U[i].transpose(), sim_time[i], 0.01)[:-1, 1:-3]
            predictions = np.vstack((predictions, pred))
        # print('diff',predictions - target)
        loss = 0.5 * np.sum(np.square(predictions - target), axis=1)
        # print(loss)
        print('error',np.mean(loss),'weights',W)
        return loss
    #
    # # res = leastsq(objective_function, W0, ftol=0.0001)
    # # print(res[0])
    #
    # # res = least_squares(objective_function, W0, bounds=([0, 0, 1.05, -np.inf], [50, 50, 1.1, np.inf]))
    res = least_squares(objective_function, W0, )
    print('res.x',res.x)
    print('res',res)

    print('Optimization finished.')

if __name__ == '__main__':
    main()