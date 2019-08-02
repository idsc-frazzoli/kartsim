#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 25.06.19 23:48

@author: mvb
"""
from itertools import product
from multiprocessing.pool import Pool

import config
import numpy as np
import os
from data_visualization.data_io import create_folder_with_time, getPKL, dataframe_from_csv
from subprocess import Popen, STDOUT
import pandas as pd

from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
from simulator.model.hybrid_lstm_model import HybridLSTMModel

import signal
import sys


def evaluate(evaluation_data_set_path, vehicle_model_type='mpc_dynamic', vehicle_model_name='', ):
    signal.signal(signal.SIGINT, signal_handler)
    visualization = False
    logging = True
    evaluation_name = vehicle_model_type + '_' + vehicle_model_name
    # evaluation_name = 'test'

    save_root_path = os.path.join(config.directories['root'], 'Evaluation')

    save_path = create_folder_with_time(save_root_path, tag=evaluation_name)
    os.mkdir(os.path.join(save_path, 'open_loop_simulation_files'))
    os.mkdir(os.path.join(save_path, 'closed_loop_simulation_files'))

    load_path = evaluation_data_set_path

    simulation_files = []
    for r, d, f in os.walk(load_path):
        for file in f:
            if '.pkl' in file:
                simulation_files.append(file)
    simulation_files.sort()

    simulate_closed_loop(vehicle_model_type, vehicle_model_name, load_path, simulation_files, save_path, visualization,
                         logging)
    simulate_open_loop(vehicle_model_type, vehicle_model_name, load_path, simulation_files, save_path)

    # save_path = os.path.join(config.directories['root'], 'Evaluation', '20190722-101020_hybrid_lstm_2x32_relu_reg0p1')
    generate_results(save_root_path, save_path, load_path)


def simulate_open_loop(vehicle_model_type, vehicle_model_name, simulation_folder, simulation_files, save_path):
    save_path = os.path.join(save_path, 'open_loop_simulation_files')
    if vehicle_model_type == 'mpc_dynamic':
        vehicle_model = DynamicVehicleMPC()

        for simulation_file in simulation_files:
            dataset_path = os.path.join(simulation_folder, simulation_file)
            # Load and batch data
            dataframe = getPKL(dataset_path)

            dataset = dataframe[
                ['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                 'steer position cal [n.a.]',
                 'brake position effective [m]', 'motor torque cmd left [A_rms]',
                 'motor torque cmd right [A_rms]']].values
            V = dataset[:, 1:4]
            U = dataset[:, -4:]
            accelerations = vehicle_model.get_accelerations(V, U)
            accelerations = np.array(accelerations).transpose()
            final_dataframe = pd.DataFrame(np.hstack((dataset, accelerations)))

            if 'mirrored' in simulation_file:
                save_file_path = os.path.join(save_path, simulation_file[:18] +
                                              '_mirrored_{}_openloop.csv'.format(vehicle_model_type))
            else:
                save_file_path = os.path.join(save_path, simulation_file[:18] +
                                              '_{}_openloop.csv'.format(vehicle_model_type))

            final_dataframe.to_csv(save_file_path, index=False,
                                   header=['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
                                           'pose vtheta [rad*s^-1]',
                                           'steer position cal [n.a.]', 'brake position effective [m]',
                                           'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]',
                                           'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                           'pose atheta [rad*s^-2]'])
    elif 'mlp' in vehicle_model_type or 'lstm' in vehicle_model_type:
        if 'mlp' in vehicle_model_type:
            vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)
        elif 'lstm' in vehicle_model_type:
            vehicle_model = HybridLSTMModel(model_name=vehicle_model_name)

        for simulation_file in simulation_files:
            dataset_path = os.path.join(simulation_folder, simulation_file)
            # Load and batch data
            dataframe = getPKL(dataset_path)

            dataset = dataframe[
                ['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                 'steer position cal [n.a.]',
                 'brake position effective [m]', 'motor torque cmd left [A_rms]',
                 'motor torque cmd right [A_rms]']].values
            # 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']].values
            V = dataset[:, 1:4]
            U = dataset[:, -4:]
            if 'mlp' in vehicle_model_type:
                accelerations = vehicle_model.get_accelerations(V, U)
            elif 'lstm' in vehicle_model_type:
                accelerations = vehicle_model.get_accelerations(0, V, U)
            # accelerations = np.array(accelerations).transpose()
            final_dataframe = pd.DataFrame(np.hstack((dataset, accelerations)))

            if 'mirrored' in simulation_file:
                save_file_path = os.path.join(save_path, simulation_file[:18] +
                                              '_mirrored_{}_openloop.csv'.format(vehicle_model_type + '_' + vehicle_model_name))
            else:
                save_file_path = os.path.join(save_path, simulation_file[:18] +
                                              '_{}_openloop.csv'.format(vehicle_model_type + '_' + vehicle_model_name))

            final_dataframe.to_csv(save_file_path, index=False,
                                   header=['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
                                           'pose vtheta [rad*s^-1]',
                                           'steer position cal [n.a.]', 'brake position effective [m]',
                                           'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]',
                                           'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                           'pose atheta [rad*s^-2]'])


def simulate_closed_loop(vehicle_model_type, vehicle_model_name, load_path, simulation_files, save_path, visualization,
                         logging):
    global client_process, visualization_process, logger_process, server_process
    client_process = visualization_process = logger_process = server_process = []
    FNULL = open(os.devnull, 'w')

    save_path = os.path.join(save_path, 'closed_loop_simulation_files')
    chunks = [simulation_files[i::8] for i in range(8)]

    argument_packages = []
    for i, chunk in enumerate(chunks):
        port = 6000 + 100 * i
        port = str(port)
        argument_packages.append([port, load_path, chunk, save_path, visualization, logging, vehicle_model_type,
                                  vehicle_model_name])
    pool = Pool(processes=8)
    pool.map(run_simulation, argument_packages)


def run_simulation(arguments):
    port, load_path, simulation_files, save_path, visualization, logging, vehicle_model_type, vehicle_model_name = arguments

    if visualization:
        vis = "1"
    else:
        vis = "0"
    if logging:
        log = "1"
    else:
        log = "0"
    print(f'Closed loop simulation started with {vehicle_model_type} {vehicle_model_name}')
    server_args = [port, vis, log, vehicle_model_type, vehicle_model_name]
    server_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/kartsim_server.py"
    serv_process = Popen(["python3", server_path] + server_args,
                         preexec_fn=os.setsid)  # , stdout=FNULL, stderr=STDOUT)
    server_process.append(serv_process)
    # print('Server started at', server_process.pid)
    if visualization:
        vis_args = [port]
        vis_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/kartsim_visualizationclient.py"
        vis_process = Popen(["python3", vis_path] + vis_args, preexec_fn=os.setsid)  # , stdout=FNULL, stderr=STDOUT)
        # print('Visualization started at', vis_process.pid)
        visualization_process.append(vis_process)

    if logging:
        log_args = [port, save_path, vehicle_model_type, vehicle_model_name] + str(simulation_files)[2:-2].split(
            '\', \'')
        log_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/kartsim_loggerclient_evaluation.py"
        log_process = Popen(["python3", log_path] + log_args,
                            preexec_fn=os.setsid)  # , stdout=FNULL, stderr=STDOUT)
        # print('Logger started at', log_process.pid)
        logger_process.append(log_process)

    client_args = [port, save_path, load_path] + str(simulation_files)[2:-2].split('\', \'')
    client_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/user/evaluationClient.py"
    cli_process = Popen(["python3", client_path] + client_args, preexec_fn=os.setsid)
    # print('Client started at', client_process.pid)
    client_process.append(cli_process)

    cli_process.communicate()
    #
    # time.sleep(10)

    # os.killpg(client_process.pid, signal.SIGTERM)
    if visualization:
        os.killpg(vis_process.pid, signal.SIGTERM)
    if logging:
        os.killpg(log_process.pid, signal.SIGTERM)
    os.killpg(serv_process.pid, signal.SIGTERM)

    print(f'{len(simulation_files)} closed loop simulations done.')


def generate_results(save_root_path, save_path, sim_folder):
    modes = [['closed_loop', 'closed_loop_simulation_files'], ['open_loop', 'open_loop_simulation_files']]

    for mode, load_folder_name in modes:
        load_path_simulation_logs = os.path.join(save_path, load_folder_name)

        simulation_files = []
        reference_files = []
        for r, d, f in os.walk(load_path_simulation_logs):
            for file in f:
                if '.csv' in file:
                    simulation_files.append([os.path.join(r, file), file])
        for r, d, f in os.walk(sim_folder):
            for file in f:
                if '.pkl' in file:
                    reference_files.append([os.path.join(r, file), file])
        simulation_files.sort()
        reference_files.sort()
        if len(simulation_files) == 0:
            continue

        grouped_files = sort_out_files(simulation_files, reference_files)
        mse_list = []
        mae_list = []
        cod_list = []
        stability_list = []
        for _, name in grouped_files[0][:-2]:
            if 'mirrored' not in name:
                model_name = name[19:-4]
                if 'unstable' in model_name:
                    model_name = model_name[:-9]
                mse_list.append([model_name, pd.DataFrame()])
                mae_list.append([model_name, pd.DataFrame()])
                cod_list.append([model_name, pd.DataFrame()])
                stability_list.append(pd.DataFrame(index=['stable_sim']))
        for files in grouped_files:
            m_reference = files.pop()
            m_log_name = m_reference[1][:18] + '_mirrored'
            m_ref_dataframe = getPKL(m_reference[0])
            m_reference_data = m_ref_dataframe[['vehicle vx [m*s^-1]',
                                                'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                                'vehicle ax local [m*s^-2]',
                                                'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
            reference = files.pop()
            log_name = reference[1][:18]
            ref_dataframe = getPKL(reference[0])
            reference_data = ref_dataframe[['vehicle vx [m*s^-1]',
                                            'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                            'vehicle ax local [m*s^-2]',
                                            'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
            for index, file_pair in enumerate(zip(files[::2], files[1::2])):
                for file in file_pair:
                    if 'unstable' in file[1]:
                        if 'mirrored' in file[1]:
                            stability_list[index][m_log_name] = np.array([0])
                            # ref = reference_data.drop(['time [s]'], axis=1)
                            empty_results = np.mean(np.square(m_reference_data))
                            for i in range(len(empty_results)):
                                empty_results[i] = np.nan
                            mse_list[index][1][m_log_name] = empty_results
                            mae_list[index][1][m_log_name] = empty_results
                            cod_list[index][1][m_log_name] = empty_results
                            continue
                        else:
                            stability_list[index][log_name] = np.array([0])
                            # ref = reference_data.drop(['time [s]'], axis=1)
                            empty_results = np.mean(np.square(reference_data))
                            for i in range(len(empty_results)):
                                empty_results[i] = np.nan
                            mse_list[index][1][log_name] = empty_results
                            mae_list[index][1][log_name] = empty_results
                            cod_list[index][1][log_name] = empty_results
                            continue
                    if 'mirrored' in file[1]:
                        stability_list[index][m_log_name] = np.array([1])
                        sim_dataframe = dataframe_from_csv(file[0])
                        try:
                            simulation_data = sim_dataframe[['vehicle vx [m*s^-1]',
                                                             'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                                             'vehicle ax local [m*s^-2]',
                                                             'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
                        except KeyError:
                            print('Empty Dataframe. Coninuing with next file')
                            continue

                        # ref = reference_data.drop(['time [s]'], axis=1)
                        # sim = simulation_data.drop(['time [s]'], axis=1)
                        error = m_reference_data - simulation_data
                        mean_squared_error = np.mean(np.square(error))
                        mean_absolute_error = np.mean(np.abs(error))
                        coeff_of_det = coeff_of_determination(m_reference_data, simulation_data)
                        mse_list[index][1][m_log_name] = mean_squared_error
                        mae_list[index][1][m_log_name] = mean_absolute_error
                        cod_list[index][1][m_log_name] = coeff_of_det
                    else:
                        stability_list[index][log_name] = np.array([1])
                        sim_dataframe = dataframe_from_csv(file[0])
                        try:
                            simulation_data = sim_dataframe[['vehicle vx [m*s^-1]',
                                                             'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                                             'vehicle ax local [m*s^-2]',
                                                             'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
                        except KeyError:
                            print('Empty Dataframe. Coninuing with next file')
                            continue

                        # ref = reference_data.drop(['time [s]'], axis=1)
                        # sim = simulation_data.drop(['time [s]'], axis=1)
                        error = reference_data - simulation_data
                        mean_squared_error = np.mean(np.square(error))
                        mean_absolute_error = np.mean(np.abs(error))
                        coeff_of_det = coeff_of_determination(reference_data, simulation_data)
                        mse_list[index][1][log_name] = mean_squared_error
                        mae_list[index][1][log_name] = mean_absolute_error
                        cod_list[index][1][log_name] = coeff_of_det
        results = pd.DataFrame()
        for (name, mse), (name, mae), (name, cod), stability in zip(mse_list, mae_list, cod_list, stability_list):
            detailed_results = pd.DataFrame()
            mean_mse = np.round(np.mean(np.mean(mse, axis=1)), 4)
            mean_mae = np.round(np.mean(np.mean(mae, axis=1)), 4)
            mean_cod = np.round(np.mean(np.mean(cod, axis=1)), 4)
            mean_stability = np.round(np.mean(np.mean(stability, axis=1)), 4)
            mse_names = ['mse_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
            mae_names = ['mae_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
            cod_names = ['cod_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
            mse.index = mse_names
            mae.index = mae_names
            cod.index = cod_names
            detailed_results = detailed_results.append(stability)
            detailed_results = detailed_results.append(mse.round(2))
            detailed_results = detailed_results.append(mae.round(2))
            detailed_results = detailed_results.append(cod.round(2))
            detailed_results = detailed_results.transpose()
            means = pd.DataFrame(np.mean(detailed_results), columns=['means'])
            detailed_results = detailed_results.append(means.transpose())
            detailed_results.to_csv(os.path.join(save_path, f'{name}_detailed_results.csv'))
            results = results.append({'Name': name, 'mean squared error': mean_mse, 'mean absolute error': mean_mae,
                                      'coefficient of determination': mean_cod, 'stability ratio': mean_stability, },
                                     ignore_index=True)
        # print(results.head())
        results.to_csv(os.path.join(save_path, 'overall_' + mode + '_results.csv'), index=False)

        with open(os.path.join(save_root_path, 'model_performances_' + mode + '.csv'), 'a') as f:
            results.to_csv(f, header=False, index=False)


def sort_out_files(sim_files, log_files):
    log_names = []
    groups = []
    for ind, (path, name) in enumerate(sim_files):
        if name[:18] not in log_names:
            log_names.append(name[:18])
            groups.append([sim_files[ind]])
            continue
        else:
            i = log_names.index(name[:18])
            groups[i].append(sim_files[ind])

    for ind, (path, name) in enumerate(log_files):
        i = log_names.index(name[:18])
        groups[i].append(log_files[ind])
    return groups


def coeff_of_determination(labels, predictions):
    total_error = np.sum(np.square(np.subtract(labels, np.mean(labels).values)))
    unexplained_error = np.sum(np.square(np.subtract(labels, predictions)))
    r_squared = np.subtract(1.0, np.divide(unexplained_error, total_error))
    return r_squared


def signal_handler(sig, frame):
    print('\nkilling all subprocesses\n')
    if len(client_process) > 0:
        for cli_process in client_process:
            # print('1')
            os.killpg(cli_process.pid, signal.SIGTERM)
    if len(visualization_process) > 0:
        for vis_process in visualization_process:
            # print('2')
            os.killpg(vis_process.pid, signal.SIGTERM)
    if len(logger_process) > 0:
        for log_process in logger_process:
            # print('3')
            os.killpg(log_process.pid, signal.SIGTERM)
    if len(server_process) > 0:
        for serv_process in server_process:
            # print('4')
            os.killpg(serv_process.pid, signal.SIGTERM)
    sys.exit(0)


if __name__ == '__main__':
    vehicle_models = [
        # ['mpc_dynamic', ''],
        ['hybrid_mlp', '5x64_relu_reg0p01_m'],
        # ['hybrid_mlp', '2x32_linear_reg0p02'],
        # ['hybrid_lstm', '2x32_relu_reg0p1'],
    ]

    for model_type, model_name in vehicle_models:
        if 'mlp' in model_type or 'mpc' in model_type:
            evaluation_data_set_name = '20190729-162122_trustworthy_mirrored'
            load_path = os.path.join(config.directories['root'], 'Data', 'MLPDatasets', evaluation_data_set_name,
                                     'test_log_files')
        elif 'lstm' in model_type:
            evaluation_data_set_name = '20190717-101005_trustworthy_data'
            # evaluation_data_set_name = '20190717-101005_trustworthy_data (copy)'
            load_path = os.path.join(config.directories['root'], 'Data', 'RNNDatasets', evaluation_data_set_name,
                                     'test_log_files')
        try:
            evaluate(load_path, model_type, model_name)
        except ValueError:
            raise
