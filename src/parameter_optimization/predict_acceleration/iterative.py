#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
from model.marcsmodel_paramopt_vectorized import \
    marc_vehiclemodel as marc_vehiclemodel_vec  # , Jacobian_marc_vehiclemodel as Jacobian_marc_vehiclemodel_vec
from model.marcsmodel_paramopt_vectorized_correct import marc_vehiclemodel as marc_vehiclemodel_vec_jcorrect
from data_visualization.data_io import getPKL, create_folder_with_time
import numpy as np

np.set_printoptions(precision=4)
import time
import datetime
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_pdf import PdfPages


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

    save_results_path = '/home/mvb/0_ETH/01_MasterThesis/OptimizationResults'
    folder_path = create_folder_with_time(save_results_path, '_iteration_test')
    file_path = folder_path + '/optimization_results.csv'

    with open(file_path, 'w') as file:
        columns_text = 'Time' + ',' + 'Optimal D1' + ',' + 'Optimal D2' + ',' + 'Optimal Ic' + ',' + 'Test loss'
        file.write(columns_text)
        file.write('\n')

    # pdffilepath = folder_path + '/loss_plots.pdf'
    # pdf = PdfPages(pdffilepath)

    """Load a data set"""
    dataframe = getPKL(dataset_path)[:50000]
    print('Training on dataset with', len(dataframe), 'samples.')

    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
                         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                         'pose atheta [rad*s^-2]']].values

    X = dataset[:, :6]
    Y = dataset[:, 6:]

    limits_D1 = [0, 10]
    limits_D2 = [5, 15]
    limits_Ic = [1, 10]

    global debug
    debug = True
    if debug:
        limits_D1 = [0, 1]
        limits_D2 = [0, 1]
        limits_Ic = [0, 2]

    resolution = 1

    total_iterations = (limits_D1[1] - limits_D1[0]) / resolution * (limits_D2[1] - limits_D2[0]) / resolution * (
                limits_Ic[1] - limits_Ic[0]) / resolution

    iteration_count = 0
    t0 = time.time()
    for D1 in range(limits_D1[0], limits_D1[1]):
        for D2 in range(limits_D2[0], limits_D2[1]):
            for Ic in range(limits_Ic[0], limits_Ic[1]):
                """Initialize parameters [D1,D2,Ic]"""
                # W0 = [7, 10, 1]  # best human guess so far
                # W0 = [9.4,10.4,0.3] #original
                # W0 = [8.0*1.1, 10.0*1.2, 1.0] #test
                W0 = [D1-0.5,D2-0.5,Ic-0.5]
                if debug:
                    W0 = [-0.5,12.5,5.5]
                    if Ic == 1:
                        W0 = [8.8,12,1]

                # print('Initialized weights to', W0)

                """Make predictions."""

                predictions, J = marc_vehiclemodel_vec_jcorrect(X, W0)

                """Calculate error and loss between prediction and measurements."""
                loss, error = loss_function(predictions, Y)

                avg_loss = np.mean(loss)

                current_dt = datetime.datetime.now()
                date_and_time = current_dt.strftime("%Y%m%d-%H%M%S")
                result_text = str(date_and_time) + ',' + str(W0[0]) + ',' + str(W0[1]) + ',' + str(W0[2]) + ',' + str(
                    avg_loss)
                with open(file_path, 'a') as file:
                    file.write(result_text)
                    file.write('\n')
                if not debug:
                    if iteration_count % int(np.round(total_iterations / 100)) == 0:
                        print('Loss:', round(avg_loss,2), 'Weights:', W0, 'Progress:', int(np.round(iteration_count / total_iterations * 100)), '%  Time elapsed:',
                              int(time.time() - t0), 's', end='\r')
                iteration_count += 1

        # fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        # fig.suptitle(date_and_time, fontsize=16)
        # # plt.figure(figsize=(10,10))
        # axs[0].scatter(range(len(loss_plot)), loss_plot, label='training loss')
        # axs[0].legend()
        # axs[1].plot(avg_loss_plot, label='average training loss for batch')
        # axs[1].legend()
        # axs[1].set_xlabel('iterations [1]')
        # pdf.savefig()
        # plt.close()
    print('Optimization finished.')
    print('Progress:', int(np.round(iteration_count / total_iterations * 100)), '%  Time elapsed:',
                          int(time.time() - t0), 's')

    # pdf.close()


def loss_function(pred, target):

    error = np.subtract(pred, target)
    squares = np.square(error)
    loss = 0.5 * np.sum(squares, axis=1)
    if debug:
        for i in range(3):
            plt.figure(i)
            plt.subplot(2,1,1)
            plt.plot(pred[:,i])
            plt.plot(target[:, i])
            plt.subplot(2, 1, 2)
            plt.plot(np.square(np.subtract(pred[:,i], target[:,i])))

            plt.show()
        print('error', np.sum(error,axis=0))
        # print('squares', loss)
        print('loss', np.sum(squares,axis=0))
    return loss, np.array(error)


if __name__ == '__main__':
    main()
