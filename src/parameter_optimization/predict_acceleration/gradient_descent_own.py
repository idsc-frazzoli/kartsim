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
    dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/Parameter_Optimization/20190517-140558_test_dataset_for_paramopt/dataset.pkl'

    save_results_path = '/home/mvb/0_ETH/01_MasterThesis/OptimizationResults'
    folder_path = create_folder_with_time(save_results_path, '_test')
    file_path = folder_path + '/optimization_results.csv'

    with open(file_path, 'w') as file:
        columns_text = 'Time' + ',' + 'Initial D1' + ',' + 'Initial D2' + ',' + 'Initial Ic' + ',' + 'Total epochs' + ',' + 'Optimal epoch' + ',' + 'Minimal training loss' + ',' + 'Test loss' + ',' + 'Optimal D1' + ',' + 'Optimal D2' + ',' + 'Optimal Ic'
        file.write(columns_text)
        file.write('\n')

    pdffilepath = folder_path + '/loss_plots.pdf'
    pdf = PdfPages(pdffilepath)

    # dataset_path = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333/20190404T133714_01_sampledlogdata.pkl'
    learning_rate = 0.0005

    """Load a data set"""
    dataframe = getPKL(dataset_path)[:50000]
    print('Training on dataset with', len(dataframe), 'samples.')

    """Create continuous batches"""
    batch_time_interval = 5  # [s]
    data_batched, no_of_batches = create_continuous_batches(dataframe, batch_time_interval)

    """Split data into train and test set"""
    random.shuffle(data_batched)
    fraction_training = 0.7
    split_point = int(len(data_batched) * fraction_training)
    training_data_batched = data_batched[:split_point]
    no_of_training_batches = len(training_data_batched)
    test_data_batched = data_batched[split_point:]
    no_of_test_batches = len(test_data_batched)

    for count in range(1):
        print('count', count)
        """Initialize parameters [D1,D2,Ic]"""
        # W0 = [7, 10, 1]  # best human guess so far
        # W0 = [9.4,10.4,0.3] #original
        # W0 = [8.0*1.1, 10.0*1.2, 1.0] #test
        limits_D1 = [5,15]
        limits_D2 = [5,15]
        limits_Ic = [0.1,10]

        W0 = [random_number_in_range(limits_D1),
             random_number_in_range(limits_D2),
             random_number_in_range(limits_Ic)] #generate random initialization weights in range

        print('Initialized weights to', W0)
        W = W0

        t0 = time.time()
        loss_plot = []
        avg_loss_plot = []
        last_avg_loss = 100
        epoch = 0
        growing_loss_counter = 0
        convergence_counter = 0

        run_optimization = True
        while run_optimization:
            # while dloss > 0.001 or tolerated_negative_loss:
            """Do parameter optimization until the improvement of the average loss is less than threshold"""
            # for epoch in range(epochs):
            #     """Create random batches"""
            #     batch_size = 100
            #     data_batched, no_of_training_batches = create_batches(dataframe, batch_size)

            """Shuffle continuous batches"""
            random.shuffle(training_data_batched)

            avg_loss = 0
            batch_no = 0
            for X, Y in training_data_batched:
                """Make predictions."""
                # predictions = []
                # for X in batch:
                #     predictions.append(marc_vehiclemodel(X[:6], W))

                # predictions, J = marc_vehiclemodel_vec(batch[:,:6], W)
                predictions, J = marc_vehiclemodel_vec_jcorrect(X, W)

                """Calculate error and loss between prediction and measurements."""
                loss, error = loss_function(predictions, Y)

                """Update weights/parameters."""
                init = True
                for j, e in zip(J, error):
                    if init:
                        update = np.matmul(j, e)
                        init = False
                    else:
                        update += np.matmul(j, e)
                    # print('matmul',np.matmul(j,e))
                update = update / len(error)

                W = W - learning_rate * update

                avg_loss += np.mean(loss) / no_of_training_batches
                if np.mean(loss) < 1000:
                    loss_plot.append(np.mean(loss))

                if batch_no % int(np.round(no_of_training_batches / 5)) == 0:
                    print('Progress:', int(np.round(batch_no / no_of_training_batches * 100)), '%      Time elapsed:',
                          time.time() - t0, 's', end='\r')
                batch_no += 1

            if len(avg_loss_plot) == 0:
                min_loss = avg_loss
                min_weights = W
                min_epoch = epoch
            elif avg_loss < min(avg_loss_plot):
                min_loss = avg_loss
                min_weights = W
                min_epoch = epoch
            avg_loss_plot.append(avg_loss)

            dloss = last_avg_loss - avg_loss
            if dloss < 0 and not growing_loss_counter > 10:
                growing_loss_counter += 1
            elif dloss < 0.001:
                convergence_counter += 1

            if growing_loss_counter > 10 or convergence_counter > 10:
                run_optimization = False

            last_avg_loss = avg_loss
            print('epoch:', epoch, 'loss:', avg_loss, 'weights:', W)
            epoch += 1

        avg_test_loss = 0
        for X_test, Y_test in test_data_batched:
            test_predictions, _ = marc_vehiclemodel_vec_jcorrect(X_test, W)
            test_loss, _ = loss_function(test_predictions, Y_test)
            avg_test_loss += np.mean(test_loss) / no_of_test_batches

        current_dt = datetime.datetime.now()
        date_and_time = current_dt.strftime("%Y%m%d-%H%M%S")
        result_text = str(date_and_time) + ',' + str(W0[0]) + ',' + str(W0[1]) + ',' + str(W0[2]) + ',' + str(epoch - 1) + ',' + str(min_epoch) + ',' + str(
            min_loss) + ',' + str(avg_test_loss) + ',' + str(min_weights[0]) + ',' + str(min_weights[1]) + ',' + str(
            min_weights[2])
        with open(file_path, 'a') as file:
            file.write(result_text)
            file.write('\n')

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(date_and_time, fontsize=16)
        # plt.figure(figsize=(10,10))
        axs[0].scatter(range(len(loss_plot)), loss_plot, label='training loss')
        axs[0].legend()
        axs[1].plot(avg_loss_plot, label='average training loss for batch')
        axs[1].legend()
        axs[1].set_xlabel('iterations [1]')
        pdf.savefig()
        plt.close()
        print('Optimization finished.')
        print('Total number of epochs:', epoch - 1, 'training loss:', avg_loss, 'test loss:', avg_test_loss, 'weights:', W)

    pdf.close()

    plt.figure(1)
    plt.scatter(range(len(loss_plot)),loss_plot)

    plt.figure(2)
    plt.plot(avg_loss_plot)
    plt.show()


def random_number_in_range(limits):
    return random.random() * (limits[1] - limits[0]) + limits[0]


def create_batches(dataframe, batch_size):
    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
                         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                         'pose atheta [rad*s^-2]']].values
    # np.random.seed(4)
    # np.random.shuffle(dataset)
    total_no_of_samples = len(dataframe)
    no_of_batches = int(total_no_of_samples / batch_size)

    data_batched = []
    for batch_no in range(no_of_batches):
        if batch_no != no_of_batches:
            X = dataset[batch_no * batch_size: (batch_no + 1) * batch_size, :6]
            Y = dataset[batch_no * batch_size: (batch_no + 1) * batch_size, 6:]
            data_batched.append([X, Y])
        else:
            X = dataset[batch_no * batch_size:, :6]
            Y = dataset[batch_no * batch_size:, 6:]
            data_batched.append([X, Y])

    return data_batched, no_of_batches


def create_continuous_batches(dataframe, batch_time_interval):
    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'MH BETA [rad]',
                         'MH AB [m*s^-2]', 'MH TV [rad*s^-2]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                         'pose atheta [rad*s^-2]']].values

    time_vals = dataframe['time [s]'].values
    dt = time_vals[1] - time_vals[0]
    interval = int(np.round(batch_time_interval / dt))
    split_rows = np.where(time_vals == 0)[0]

    data_batched = []
    no_of_batches = 0
    for start, stop in zip(split_rows, np.append(split_rows[1:], len(time_vals))):
        for i in range(int(np.floor((stop - start) / interval))):
            # print(' ',start + i * interval , start + (i + 1) * interval)
            chunk = dataset[start + i * interval: start + (i + 1) * interval]
            X = chunk[:, :6]
            Y = chunk[:, 6:]
            data_batched.append([X, Y])
            no_of_batches += 1

        # print(' ',start + int(np.floor((stop-start)/interval)) * interval, stop)
        chunk = dataset[start + int(np.floor((stop - start) / interval)) * interval: stop]
        X = chunk[:, :6]
        Y = chunk[:, 6:]
        data_batched.append([X, Y])
        no_of_batches += 1

    return data_batched, no_of_batches


def loss_function(pred, target):
    error = np.subtract(pred, target)
    squares = np.square(error)
    loss = 0.5 * np.sum(squares, axis=1)
    return loss, np.array(error)


if __name__ == '__main__':
    main()
