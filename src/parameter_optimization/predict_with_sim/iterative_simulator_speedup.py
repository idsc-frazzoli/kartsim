#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
import parameter_optimization.integrate.timeIntegrators as integrators
from dataanalysisV2.dataIO import getPKL, create_folder_with_time
import numpy as np
import time
import datetime
import multiprocessing
import matplotlib.pyplot as plt
from random import shuffle

def main():
    global file_path, iteration_count, total_iterations, t0, results
    np.set_printoptions(precision=4)

    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190515-133222_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-111946_test_dataset_for_paramopt/dataset.pkl'
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'
    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'

    save_results_path = '/home/mvb/0_ETH/01_MasterThesis/OptimizationResults'
    folder_path = create_folder_with_time(save_results_path, '_iterative_simulator_test')
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
    param_space_D1 = np.linspace(5, 14, num=10)
    param_space_D2 = np.linspace(9, 19, num=10)
    param_space_Ic = np.linspace(0.6, 2.0, num=10)

    total_iterations = len(param_space_D1) * len(param_space_D2) * len(param_space_Ic)
    print("Total number of iterations:", total_iterations)

    weight_list = []
    for D1 in param_space_D1:
        for D2 in param_space_D2:
            for Ic in param_space_Ic:
                weight_list.append([D1, D2, Ic])

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    t0 = time.time()
    iteration_count = 0
    results = []
    for W0 in weight_list:
        pool.apply_async(simulate_for_weights, args=(W0, data_batched, no_of_batches), callback=write_results_to_file)

    pool.close()
    pool.join()
    print('Optimization finished.')
    print('Progress: 100 % Time elapsed:', int(time.time()-t0),'s')


def simulate_for_weights(W0, data_batched, no_of_batches):
    avg_loss = 0

    t1 = time.time()
    for X0, sim_time, U, target in data_batched:
        XW = np.append(X0, W0)

        pred = integrators.odeIntegratorIVP(XW, U.transpose(), sim_time, 0.01)[:-1, 1:-3]

        loss, error = loss_function(pred[:-1, :], target)

        avg_loss += loss / no_of_batches
    print('overall', time.time()-t1)

    return (W0, avg_loss)

def write_results_to_file(result):
    global iteration_count, results
    W0, avg_loss = result
    current_dt = datetime.datetime.now()
    date_and_time = current_dt.strftime("%Y%m%d-%H%M%S")
    result_text = str(date_and_time) + ',' + str(W0[0]) + ',' + str(W0[1]) + ',' + str(W0[2]) + ',' + str(
        avg_loss[0]) + ',' + str(avg_loss[1]) + ',' + str(avg_loss[2]) + ',' + str(avg_loss[3]) + ',' + str(
        avg_loss[4]) + ',' + str(avg_loss[5]) + ',' + str(np.mean(avg_loss))
    # results.append(result_text)

    with open(file_path, 'a') as file:
        file.write(result_text)
        file.write('\n')

    iteration_count += 1
    if iteration_count % int(np.round(total_iterations / 100)) == 0:
        print('Weights:', W0, 'Progress:', int(np.round(iteration_count / total_iterations * 100)), '%  Time elapsed:',
              int(time.time() - t0), 's', end='\r')



def loss_function(pred, target):
    error = np.subtract(pred, target)
    squares = np.square(error)
    loss = 0.5*np.mean(squares,axis=0)
    return loss, np.array(error)

if __name__ == '__main__':
    main()