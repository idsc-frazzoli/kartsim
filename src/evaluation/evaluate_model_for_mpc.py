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
from simulator.model.kinematic_mpc_model import KinematicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
from simulator.model.hybrid_lstm_model import HybridLSTMModel
from simulator.model.no_model_model import NoModelModel
from simulator.model.no_model_sparse_model import NoModelSparseModel

import signal
import sys


def evaluate(evaluation_data_set_path, vehicle_model_type='mpc_dynamic', vehicle_model_name='', ):
    signal.signal(signal.SIGINT, signal_handler)
    visualization = False
    logging = True
    evaluation_name = vehicle_model_type + '_' + vehicle_model_name + '_mpcinputs'
    # evaluation_name = 'testk'
    if '(copy)' in evaluation_data_set_path:
        evaluation_name += '_single'

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

    # save_path = os.path.join(config.directories['root'], 'Evaluation', '20190919-084329_hybrid_mlp_0x6_None_reg0p001_dyn_directinput_fulldata_mlpsymmetric_mpcinputs')
    generate_results(save_root_path, save_path, load_path)


def simulate_open_loop(vehicle_model_type, vehicle_model_name, simulation_folder, simulation_files, save_path):
    save_path = os.path.join(save_path, 'open_loop_simulation_files')
    if vehicle_model_type in ['mpc_dynamic', 'mpc_kinematic', 'mlp', 'no_model', 'no_model_sparse']:
        if vehicle_model_type == 'mpc_dynamic':
            vehicle_model = DynamicVehicleMPC(direct_input=True)
        elif vehicle_model_type == 'mpc_kinematic':
            vehicle_model = KinematicVehicleMPC(direct_input=True)
        elif vehicle_model_type == 'mlp' or vehicle_model_type == 'no_model':
            vehicle_model = NoModelModel(direct_input=True, model_name=vehicle_model_name)
        elif vehicle_model_type == 'no_model_sparse':
            vehicle_model = NoModelSparseModel(model_name=vehicle_model_name)

        for simulation_file in simulation_files:
            dataset_path = os.path.join(simulation_folder, simulation_file)
            # Load and batch data
            dataframe = getPKL(dataset_path)

            dataset = dataframe[
                ['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                 'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                 'acceleration torque vectoring [rad*s^-2]']].values
            V = dataset[:, 1:4]
            U = dataset[:, -3:]
            accelerations = vehicle_model.get_accelerations(V, U)
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
                                           'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                                           'acceleration torque vectoring [rad*s^-2]',
                                           'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                           'pose atheta [rad*s^-2]'])
    elif 'mlp' in vehicle_model_type or 'lstm' in vehicle_model_type:
        if 'mlp' in vehicle_model_type:
            vehicle_model = DataDrivenVehicleModel(model_type=vehicle_model_type, model_name=vehicle_model_name, direct_input=True)
        # elif 'lstm' in vehicle_model_type:
        #     vehicle_model = HybridLSTMModel(model_name=vehicle_model_name)

        for simulation_file in simulation_files:
            dataset_path = os.path.join(simulation_folder, simulation_file)
            # Load and batch data
            dataframe = getPKL(dataset_path)

            dataset = dataframe[
                ['time [s]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                 'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                 'acceleration torque vectoring [rad*s^-2]']].values
            V = dataset[:, 1:4]
            U = dataset[:, -3:]
            accelerations = vehicle_model.get_accelerations(V, U)
            # elif 'lstm' in vehicle_model_type:
            #     accelerations = vehicle_model.get_accelerations(0, V, U)
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
                                           'turning angle [n.a]', 'acceleration rear axle [m*s^-2]',
                                           'acceleration torque vectoring [rad*s^-2]',
                                           'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                           'pose atheta [rad*s^-2]'])


def simulate_closed_loop(vehicle_model_type, vehicle_model_name, load_path, simulation_files, save_path, visualization,
                         logging):
    global client_process, visualization_process, logger_process, server_process
    client_process = visualization_process = logger_process = server_process = []
    FNULL = open(os.devnull, 'w')

    save_path = os.path.join(save_path, 'closed_loop_simulation_files')
    if len(simulation_files) >= 5:
        chunks = [simulation_files[i::5] for i in range(5)]
        argument_packages = []
        for i, chunk in enumerate(chunks):
            port = 6000 + 100 * i
            port = str(port)
            argument_packages.append([port, load_path, chunk, save_path, visualization, logging, vehicle_model_type,
                                      vehicle_model_name])
        pool = Pool(processes=5)
        pool.map(run_simulation, argument_packages)
    elif len(simulation_files) > 1:
        no_settings = len(simulation_files)
        chunks = [simulation_files[i::no_settings] for i in range(no_settings)]
        argument_packages = []
        for i, chunk in enumerate(chunks):
            port = 6000 + 100 * i
            port = str(port)
            argument_packages.append([port, load_path, chunk, save_path, visualization, logging, vehicle_model_type,
                                      vehicle_model_name])
        pool = Pool(processes=no_settings)
        pool.map(run_simulation, argument_packages)
    else:
        port = 6000
        port = str(port)
        argument = [port, load_path, simulation_files, save_path, visualization, logging, vehicle_model_type,
                    vehicle_model_name]
        run_simulation(argument)


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
    server_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/kartsim_server_for_mpc.py"
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
        log_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/kartsim_loggerclient_for_mpc.py"
        log_process = Popen(["python3", log_path] + log_args,
                            preexec_fn=os.setsid)  # , stdout=FNULL, stderr=STDOUT)
        # print('Logger started at', log_process.pid)
        logger_process.append(log_process)

    client_args = [port, save_path, load_path] + str(simulation_files)[2:-2].split('\', \'')
    client_path = "/home/mvb/0_ETH/01_MasterThesis/kartsim/src/simulator/user/evaluationClient_for_mpc.py"
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

        model_name = grouped_files[0][0][1][19:-4]
        if 'unstable' in model_name:
            model_name = model_name[:-9]
        if 'mirrored' in model_name:
            model_name = model_name[9:]
        mse = pd.DataFrame()
        mae = pd.DataFrame()
        cod = pd.DataFrame()
        stability = pd.DataFrame(index=['stable_sim'])
        for files in grouped_files:
            reference = files.pop()
            if 'mirrored' in reference[1]:
                log_name = reference[1][:27]
            else:
                log_name = reference[1][:18]
            ref_dataframe = getPKL(reference[0])
            reference_data = ref_dataframe[['vehicle vx [m*s^-1]',
                                            'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                            'vehicle ax local [m*s^-2]',
                                            'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
            if 'unstable' in files[0][1]:
                stability[log_name] = np.array([0])
                empty_results = np.mean(np.square(reference_data))
                for i in range(len(empty_results)):
                    empty_results[i] = np.nan
                mse[log_name] = empty_results
                mae[log_name] = empty_results
                cod[log_name] = empty_results
                continue

            else:
                stability[log_name] = np.array([1])
                sim_dataframe = dataframe_from_csv(files[0][0])
                try:
                    simulation_data = sim_dataframe[['vehicle vx [m*s^-1]',
                                                     'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                                     'vehicle ax local [m*s^-2]',
                                                     'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]']]
                except KeyError:
                    print('Empty Dataframe. Coninuing with next file')
                    continue

                error = reference_data - simulation_data
                mean_squared_error = np.mean(np.square(error))
                mean_absolute_error = np.mean(np.abs(error))
                coeff_of_det = coeff_of_determination(reference_data, simulation_data)
                mse[log_name] = mean_squared_error
                mae[log_name] = mean_absolute_error
                cod[log_name] = coeff_of_det
        results = pd.DataFrame()
        # for mse, mae, cod, stability in zip(mse, mae, cod, stability):
        detailed_results = pd.DataFrame()
        mse_names = ['mse_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
        mae_names = ['mae_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
        cod_names = ['cod_' + ''.join([s.split(' ')[1], s.split(' ')[-1]]) for s in list(mse.index)]
        mse.index = mse_names
        mae.index = mae_names
        cod.index = cod_names
        detailed_results = detailed_results.append(stability)
        detailed_results = detailed_results.append(mse.round(5))
        detailed_results = detailed_results.append(mae.round(5))
        detailed_results = detailed_results.append(cod.round(5))
        detailed_results = detailed_results.transpose()
        if mode == 'open_loop':
            for name_list in [mse_names, mae_names, cod_names]:
                for topic in name_list[:3]:
                    detailed_results.pop(topic)
        means = pd.DataFrame(np.mean(detailed_results), columns=['means']).round(5)
        detailed_results = detailed_results.append(means.transpose())
        detailed_results.to_csv(os.path.join(save_path, f'{model_name}_detailed_results.csv'))

        means.columns = [os.path.basename(os.path.normpath(save_path))]
        results = results.append(means.transpose())
        results.to_csv(os.path.join(save_path, 'overall_' + mode + '_results.csv'), index=True)

        with open(os.path.join(save_root_path, 'model_performances_' + mode + '.csv'), 'a') as f:
            results.to_csv(f, header=False, index=True)


def sort_out_files(sim_files, log_files):
    log_names = []
    groups = []
    for ind, (path, name) in enumerate(sim_files):
        if 'mirrored' in name:
            name = name[:27]
        else:
            name = name[:18]
        if name not in log_names:
            log_names.append(name)
            groups.append([sim_files[ind]])
            continue
        else:
            i = log_names.index(name)
            groups[i].append(sim_files[ind])

    for ind, (path, name) in enumerate(log_files):
        if 'mirrored' in name:
            name = name[:27]
        else:
            name = name[:18]
        i = log_names.index(name)
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
        # ['mpc_kinematic', ''],
        # ['hybrid_mlp', '0x6_None_reg0p0001_directinput'],
        # ['hybrid_mlp', '0x6_None_reg0p001_directinput'],
        # ['hybrid_mlp', '0x6_None_reg1_directinput'],

        # ['no_model_sparse', 'poly3reduced_sparse'],
        # ['hybrid_kinematic_mlp', '1x16_tanh_reg0p0_kin_directinput_mlpsymmetric_detailed'],
        # ['hybrid_kinematic_mlp', '1x24_tanh_reg0p0_kin_directinput_mlpsymmetric_detailed'],
        # ['hybrid_kinematic_mlp', '1x24_softplus_reg0p0_kin_directinput_mlpsymmetric_detailed'],
        # ['hybrid_kinematic_mlp', '1x16_softplus_reg0p0_kin_directinput_mlpsymmetric_detailed'],
        # ['no_model', '1x24_tanh_reg0p0_nomodel_directinput_mlpsymmetric_detailed'],
        # ['no_model', '1x16_tanh_reg0p0_nomodel_directinput_mlpsymmetric_detailed'],
        # ['no_model', '1x16_softplus_reg0p0_nomodel_directinput_mlpsymmetric_detailed'],
        # ['no_model', '1x24_softplus_reg0p0_nomodel_directinput_mlpsymmetric_detailed'],

        ['hybrid_mlp', '0x6_None_reg0p01_dyn_directinput_fulldata_mlpsymmetric'],
        ['hybrid_mlp', '1x16_tanh_reg0p01_dyn_directinput_fulldata_mlpsymmetric'],


    ]

    for model_type, model_name in vehicle_models:
        if 'mlp' in model_type or 'mpc' in model_type or 'no_model' in model_type:
            # evaluation_data_set_name = '20190829-091021_final_data_set_dynamic_directinput'
            # evaluation_data_set_name = '20190829-091514_final_data_set_kinematic_directinput'
            # evaluation_data_set_name = '20190829-091514_final_data_set_kinematic_directinput (copy)'
            evaluation_data_set_name = '20190829-092236_final_data_set_nomodel_directinput'
            load_path = os.path.join(config.directories['root'], 'Data', 'MLPDatasets', evaluation_data_set_name,
                                     'test_log_files')
        # elif 'lstm' in model_type:
        #     evaluation_data_set_name = '20190717-101005_trustworthy_data'
        #     # evaluation_data_set_name = '20190717-101005_trustworthy_data (copy)'
        #     load_path = os.path.join(config.directories['root'], 'Data', 'RNNDatasets', evaluation_data_set_name,
        #                              'test_log_files')
        try:
            evaluate(load_path, model_type, model_name)
        except ValueError:
            raise
