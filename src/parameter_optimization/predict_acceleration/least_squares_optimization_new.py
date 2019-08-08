#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.05.19 15:26

@author: mvb
"""
# from parameter_optimization.model.marcsmodel_paramopt_scipy import marc_vehiclemodel
# from parameter_optimization.model.marcsmodel_paramopt_vectorized import marc_vehiclemodel as marc_vehiclemodel_vec
from parameter_optimization.model.marcsmodel_paramopt_new import marc_vehiclemodel as marc_vehiclemodel_vec_jcorr
from data_visualization.data_io import getPKL, data_to_pkl
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
import numpy as np
import pandas as pd
import datetime
from multiprocessing.pool import Pool

np.set_printoptions(precision=4)
import time
from scipy.optimize import least_squares, leastsq
import os


def main():
    global dataset
    path_data_set = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Data/Sampled/20190806-184249_trustworthy_paramopt_data'

    dataframe = getPKL(os.path.join(path_data_set, 'merged_sampledlogdata.pkl'))
    # dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
    #               'turning angle [n.a]', 'acceleration rear axle [m*s^-2]', 'acceleration torque vectoring [rad*s^-2]',
    #               'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
    dataset = dataframe[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                         'steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]',
                         'motor torque cmd right [A_rms]',
                         'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
    dataset = get_mpc_inputs(dataset)
    dataset = dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                         'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                         'acceleration torque vectoring [rad*s^-2]',
                         'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].values
    # Initialize parameters
    # W = [1,1,1,1]
    # W0 = np.array([9.4, 10.4, 0.3])
    # W0 = np.array([9, 1, 10, 5.2, 1.1, 10, 0.3])
    # W0 = np.random.rand(6) * 20.0
    # W0 = np.concatenate((W0, np.random.rand(1) * 2.0))
    # np.random.shuffle(dataset)
    #
    # X = dataset

    W0s = [np.array([9, 1, 10, 5.2, 1.1, 10, 0.3])]
    run_optimization(W0s)
    # n_pool = 5
    # repeat_times = 4
    # W0s = []
    # for i in range(n_pool):
    #     for j in range(repeat_times):
    #         W0 = np.random.rand(6) * 20.0
    #         # W0 = np.concatenate((W0, np.random.rand(1) * 0.3))
    #         W0 = np.concatenate((W0, np.random.rand(1) * 2.0))
    #         W0s.append(W0)
    # chunks = [W0s[i::n_pool] for i in range(n_pool)]
    # pool = Pool(processes=n_pool)
    # pool.map(run_optimization, chunks)


def run_optimization(W0s):
    for W0 in W0s:
        global loss_min
        loss_min = 1000
        np.random.shuffle(dataset)
        X = dataset
        print('Optimization running with initialization ', W0, '...')
        res = least_squares(objective_function, W0, args=(X,), ftol=1e-4, bounds=([0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 2]))
        # res = least_squares(objective_function, W0, args=(X,), ftol=1e-4, bounds=([0], [10]))
        results = pd.DataFrame()
        # results['offset_vx_0'] = [round(W0[0],4)]
        # results['offset_vy_0'] = [round(W0[1],4)]
        # results['offset_vtheta_0'] = [round(W0[2],4)]
        results['B10'] = [round(W0[0],4)]
        results['C10'] = [round(W0[1],4)]
        results['D10'] = [round(W0[2],4)]
        results['B20'] = [round(W0[3],4)]
        results['C20'] = [round(W0[4],4)]
        results['D20'] = [round(W0[5],4)]
        results['I0'] = [round(W0[6],4)]
        # results['offset_vx'] = [round(W_best[0],4)]
        # results['offset_vy'] = [round(W_best[1],4)]
        # results['offset_vtheta'] = [round(W_best[2],4)]
        results['B1'] = [round(W_best[0],4)]
        results['C1'] = [round(W_best[1],4)]
        results['D1'] = [round(W_best[2],4)]
        results['B2'] = [round(W_best[3],4)]
        results['C2'] = [round(W_best[4],4)]
        results['D2'] = [round(W_best[5],4)]
        results['I'] = [round(W_best[6],4)]
        results['error ax'] = [round(loss_min[0],4)]
        results['error ay'] = [round(loss_min[1],4)]
        results['error atheta'] = [round(loss_min[2],4)]
        results['error sum'] = [round(np.sum(loss_min),4)]

        current_dt = datetime.datetime.now()
        date_time = current_dt.strftime("%Y%m%d-%H%M%S")
        results.index = [date_time]
        save_root_path = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/parameter_optimization'
        with open(os.path.join(save_root_path, 'optimization_results.csv'), 'a') as f:
            results.to_csv(f, header=False, index=True)
        print('Optimization', np.round(W0,1), 'finished with score', round(np.sum(loss_min),4))

def objective_function(W, X):
    global loss, loss_min, W_best
    pred = marc_vehiclemodel_vec_jcorr(X[:, :6], W)
    # loss = 0.5 * np.mean(np.square(pred - X[:, 6:]), axis=0)
    loss = np.sum(np.abs(pred - X[:, 6:]), axis=0)
    if np.sum(loss) < np.sum(loss_min):
        loss_min = loss
        W_best = W
    print('error', round(np.sum(loss),4), 'loss', np.round(loss,4), 'weights', np.round(W,4))
    return loss
def get_mpc_inputs(dataframe):
    dataframe_transformed = transform_inputs(dataframe)
    return dataframe_transformed


def transform_inputs(dataframe):
    turning_angle, acceleration_rear_axle, torque_tv = DynamicVehicleMPC().transform_inputs(
        dataframe['steer position cal [n.a.]'].values,
        dataframe['brake position effective [m]'].values,
        dataframe['motor torque cmd left [A_rms]'].values,
        dataframe['motor torque cmd right [A_rms]'].values,
        dataframe['vehicle vx [m*s^-1]'].values,
    )
    dataframe['turning angle [n.a]'] = turning_angle
    dataframe['acceleration rear axle [m*s^-2]'] = acceleration_rear_axle
    dataframe['acceleration torque vectoring [rad*s^-2]'] = torque_tv
    return dataframe

if __name__ == '__main__':
    main()
