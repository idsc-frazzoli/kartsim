#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
import parameter_optimization.integrate.timeIntegrators as integrators
from data_visualization.data_io import getPKL, create_folder_with_time
import numpy as np
import time
import datetime

import matplotlib.pyplot as plt
from random import shuffle

def main():
    np.set_printoptions(precision=4)

    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190515-133222_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-111946_test_dataset_for_paramopt/dataset.pkl'
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'

    save_results_path = '/home/mvb/0_ETH/01_MasterThesis/OptimizationResults'
    folder_path = create_folder_with_time(save_results_path, '_iterative_simulator_0')
    file_path = folder_path + '/optimization_results.csv'

    with open(file_path, 'w') as file:
        columns_text = 'Time' + ',' + 'Optimal D1' + ',' + 'Optimal D2' + ',' + 'Optimal Ic' + ',' + 'pose x loss' + ',' + 'pose y loss' + ',' + 'pose theta loss' + ',' + 'vehicle v_x loss' + ',' + 'vehicle v_y loss' + ',' + 'vehicle v_theta loss' + ',' + 'Average loss'
        file.write(columns_text)
        file.write('\n')

    validation_horizon = 1.0

    # Load and batch data
    dataframe = getPKL(dataset_path)
    total_no_of_samples = len(dataframe)
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
    no_of_batches = 0
    for start, stop in zip(split_rows, np.append(split_rows[1:], len(time_vals))):
        for i in range(int(np.floor((stop - start) / interval))):
            # print(' ',start + i * interval , start + (i + 1) * interval)
            chunk = dataset[start + i * interval: start + (i + 1) * interval]
            X0 = chunk[0, :7]
            sim_time = validation_horizon
            U = np.hstack((chunk[:, 0:1], chunk[:, 10:]))
            target = chunk[:-1, 1:7]
            data_batched.append([X0,sim_time,U,target])
            no_of_batches += 1

        # print(' ',start + int(np.floor((stop-start)/interval)) * interval, stop)
        chunk = dataset[start + int(np.floor((stop - start) / interval)) * interval: stop]
        X0 = chunk[0, :7]
        sim_time = chunk[-1, 0] - chunk[0, 0]
        U = np.hstack((chunk[:, 0:1], chunk[:, 10:]))
        target = chunk[:-2, 1:7]
        data_batched.append([X0, sim_time, U, target])
        no_of_batches += 1

    # Initialize parameters [D1,D2,Ic]
    limits_D1 = [5, 15]
    limits_D2 = [5, 15]
    limits_Ic = [1, 10]

    global debug
    debug = False
    if debug:
        limits_D1 = [0, 1]
        limits_D2 = [0, 1]
        limits_Ic = [0, 2]

    resolution = 1

    total_iterations = (limits_D1[1] - limits_D1[0]) / resolution * (limits_D2[1] - limits_D2[0]) / resolution * (
            limits_Ic[1] - limits_Ic[0]) / resolution

    iteration_count = 0
    t0 = time.time()

    #speed up

    for D1 in range(limits_D1[0], limits_D1[1]):
        for D2 in range(limits_D2[0], limits_D2[1]):
            for Ic in range(limits_Ic[0], limits_Ic[1]):

                """Initialize parameters [D1,D2,Ic]"""
                # W0 = [7, 10, 1]  # best human guess so far
                # W0 = [9.4,10.4,0.3] #original
                # W0 = [8.0*1.1, 10.0*1.2, 1.0] #test
                W0 = [D1, D2, Ic]
                if debug:
                    W0 = [-0.5, 12.5, 5.5]
                    # W0 = [1, 1, 1, 1]
                    # W0 = [9.4, 10.4, 0.3]
                    # if Ic == 1:
                    #     W0 = [8.8, 12, 1]

                batch_no = 0
                avg_loss = 0
                t1 = time.time()
                for X0, sim_time, U, target in data_batched:
                    XW = np.append(X0, W0)

                    pred = integrators.odeIntegratorIVP(XW, U.transpose(), sim_time, 0.01)[:-1, 1:-3]

                    loss, error = loss_function(pred[:-1, :], target)

                    avg_loss += loss / no_of_batches
                print('time', time.time()-t1)
                current_dt = datetime.datetime.now()
                date_and_time = current_dt.strftime("%Y%m%d-%H%M%S")
                result_text = str(date_and_time) + ',' + str(W0[0]) + ',' + str(W0[1]) + ',' + str(W0[2]) + ',' + str(
                    avg_loss[0]) + ',' + str(avg_loss[1]) + ',' + str(avg_loss[2]) + ',' + str(avg_loss[3]) + ',' + str(
                    avg_loss[4]) + ',' + str(avg_loss[5]) + ',' + str(np.mean(avg_loss))
                with open(file_path, 'a') as file:
                    file.write(result_text)
                    file.write('\n')
                # if not debug:
                #     if iteration_count % int(np.round(total_iterations / 100)) == 0:
                #         print('Average loss:', round(np.mean(avg_loss), 2), 'Weights:', W0, 'Progress:',
                #               int(np.round(iteration_count / total_iterations * 100)), '%  Time elapsed:',
                #               int(time.time() - t0), 's', end='\r')
                iteration_count += 1

    print('Optimization finished.')
    print('Progress:', int(np.round(batch_no/no_of_batches*100)),'% Time elapsed:', int(time.time()-t0),'s')


def loss_function(pred, target):
    if len(pred)<len(target):
        target = target[:-(len(target)-len(pred))]
    elif len(pred)>len(target):
        pred = pred[:-(len(pred)-len(target))]

    error = np.subtract(pred, target)
    squares = np.square(error)
    loss = 0.5*np.mean(squares,axis=0)
    return loss, np.array(error)

if __name__ == '__main__':
    main()