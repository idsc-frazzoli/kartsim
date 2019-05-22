#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
from model.marcsmodel_paramopt_vectorized import Jacobian_marc_vehiclemodel as Jacobian_marc_vehiclemodel_vec
import parameter_optimization.integrate.timeIntegrators as integrators
from dataIO import getPKL
import numpy as np
np.set_printoptions(precision=4)
import time
import matplotlib.pyplot as plt
from random import shuffle

# #__for profiler
# import pandas as pd
# def getPKL(filePath):
#     dataFrame = pd.read_pickle(str(filePath))
#     return dataFrame

def main():
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190515-133222_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-111946_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'

    batch_size = 100
    learning_rate = 0.0001
    epochs = 1
    validation_horizon = 1.0

    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_no_of_samples = len(dataframe)
    no_of_batches = int(total_no_of_samples / batch_size)
    print('Training on dataset with', total_no_of_samples, 'samples.')

    dataset = dataframe[['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]',
                         'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]',
                         'vehicle ay local [m*s^-2]',
                         'pose atheta [rad*s^-2]', 'MH BETA [rad]', 'MH AB [m*s^-2]', 'MH TV [rad*s^-2]']].values

    time_vals = dataframe['time [s]'].values
    dt = time_vals[1] - time_vals[0]
    interval = int(np.round(validation_horizon / dt))

    split_rows = np.where(time_vals == 0)[0]

    data_batched = []
    for start, stop in zip(split_rows, np.append(split_rows[1:], len(time_vals))):
        for i in range(int(np.floor((stop - start) / interval))):
            # print(' ',start + i * interval , start + (i + 1) * interval)
            chunk = dataset[start + i * interval: start + (i + 1) * interval]
            X0 = chunk[0, :7]
            sim_time = validation_horizon
            U = np.hstack((chunk[:, 0:1], chunk[:, 10:]))
            target = chunk[:-1, 1:7]
            data_batched.append([X0,sim_time,U,target])


        # print(' ',start + int(np.floor((stop-start)/interval)) * interval, stop)
        chunk = dataset[start + int(np.floor((stop - start) / interval)) * interval: stop]
        X0 = chunk[0, :7]
        sim_time = chunk[-1, 0] - chunk[0, 0]
        U = np.hstack((chunk[:, 0:1], chunk[:, 10:]))
        target = chunk[:-1, 1:7]
        data_batched.append([X0, sim_time, U, target])

    # Initialize parameters [D1,D2,Ic]
    # W = [1,1,1,1]
    W = [9.4,10.4,0.3]
    t0 = time.time()
    loss_plot = []
    for epoch in range(epochs):
        # Shuffle dataset
        # shuffle(data_batched)

        avg_loss = 0
        for X0, sim_time, U, target in data_batched:

            XW = np.append(X0, W)
            pred = integrators.odeIntegratorIVP(XW, U.transpose(), sim_time, 0.01)[:-1, 1:-3]
            # print(pred[-1])

            XU = np.hstack((target[:,-3:],U[:-1,-3:]))

            J = Jacobian_marc_vehiclemodel_vec(XU, W)

            # print('pred',predictions)
            # print('target',batch[:,6:])

            loss, error = loss_function(pred[:-1,:], target)
            # print('loss',np.mean(loss))
            # print('error',error)

            # J = Jacobian_marc_vehiclemodel(X[:6], W)
            # J = Jacobian_marc_vehiclemodel_vec(batch[:,:6], W)

            init = True
            for j,e in zip(J,error):
                print(j.shape)
                print(e.shape)
                if init:
                    update = np.matmul(j,e)
                    init = False
                else:
                    update += np.matmul(j,e)
                # print('matmul',np.matmul(j,e))
            update = update / len(error)
            print('update',update)
    #
    #         # W = W - learning_rate * np.matmul(J.transpose(), error)
    #         W = W - learning_rate * update
    #
    #         avg_loss += np.mean(loss) / no_of_batches
    #         if np.mean(loss) < 1000:
    #             # print('predictions-target',predictions-batch[:,6:])
    #             loss_plot.append(np.mean(loss))
    #
    #         if batch_no % int(np.round(no_of_batches/20)) == 0:
    #             print('Progress:', int(np.round(batch_no/no_of_batches*100)),'%      Time elapsed:', time.time()-t0,'s',end='\r')
    #
    #     if epoch % 1 == 0:
    #         print('epoch:', epoch, 'loss:', avg_loss, 'weights:', W)
    # print('Optimization finished.')
    # print('epoch:', epoch, 'loss:', avg_loss, 'weights:', W)
    #
    # plt.plot(loss_plot)
    # window_len = batch_size*5
    # s = np.r_[loss_plot[window_len - 1:0:-1], loss_plot, loss_plot[-2:-window_len - 1:-1]]
    # w=np.hanning(window_len)
    # plt.plot(np.convolve(w/w.sum(),s,mode='valid'))
    # plt.show()


def loss_function(pred, target):
    error = np.subtract(pred, target)
    squares=np.square(error)
    loss = 0.5*np.sum(squares,axis=1)
    return loss, np.array(error)

if __name__ == '__main__':
    main()