#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import threading
import numpy as np
import time
import os
import pickle
import pandas as pd
import sys

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt


def main():
    # ___user inputs
    port = int(sys.argv[1])
    logging = int(sys.argv[2])
    pathsavedata = sys.argv[3]
    pathpreprodata = sys.argv[4]
    preprofiles = sys.argv[5:]

    # Choose whether to use simulation over intervals of the duration given by validationhorizon
    validation = False
    validationhorizon = 1  # [s] time inteval after which initial conditions are reset to values from log data

    # Simulation time after which result is returned from server
    server_return_interval = 2  # [s]
    # DO NOT CHANGE! Parameter used for real time simulation (default:0)
    _wait_for_real_time = 0  # [s]

    # Choose whether to simulate in real-time (mainly for visualization purposes)
    real_time = False
    real_time_factor = 2
    if real_time:
        server_return_interval = 0.1 * real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval * 0.9 * (1.0 / real_time_factor)

    # Connect to simulation server (kartsim_server.py)
    connected = False
    address = ('localhost', port)
    while not connected:
        try:
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            # print('ConnectionRefusedError')
            pass

    # generate simulation info file and store it in target folder
    if logging:
        with open(pathsavedata + '/' + 'simulationinfo.txt', 'a') as the_file:
            if validation:
                the_file.write('simulation mode:                    closed loop intervals' + '\n')
                the_file.write('initial condition reset interval:   ' + str(validationhorizon) + 's\n')
            else:
                the_file.write('simulation mode:                    closed loop' + '\n')
            the_file.write('source folder:                      ' + pathpreprodata + '\n')
            the_file.write('simulation time step:               ' + str(server_return_interval) + 's\n')
    tgo = time.time()

    for file_number, fileName in enumerate(preprofiles):
        filePath = pathpreprodata + '/' + fileName
        try:
            with open(filePath, 'rb') as f:
                preprodata = pickle.load(f)
        except:
            print('Could not open file at', filePath)
            preprodata = pd.DataFrame()
            raise

        # ___simulation parameters
        # [s] Log data sampling period
        data_time_step = np.round(preprodata['time [s]'].iloc[1] - preprodata['time [s]'].iloc[0], 3)  # [s]
        # [s] time increment used in integration method inside simulation server (time step for logged simulation data)
        sim_time_increment = data_time_step
        # [s] Total simulation time
        sim_time = preprodata['time [s]'].iloc[-1]

        # ______^^^______

        # print(f'Total time: {int(time.time() - tgo)}s. Simulation with file {file_number}/{len(preprofiles)} {fileName} started...')
        t_start = time.time()
        outcome = execute_simulation(conn, _wait_for_real_time, sim_time, sim_time_increment, server_return_interval,
                           data_time_step, validation, validationhorizon, preprodata)
        if 'finished' in outcome:
            print(f'Total time: {int(time.time() - tgo)}s {file_number + 1}/{len(preprofiles)} {fileName} successful after {int(time.time() - t_start)}s')
        elif 'aborted' in outcome:
            print(f'Total time: {int(time.time() - tgo)}s {file_number + 1}/{len(preprofiles)} {fileName} timed out after {int(time.time() - t_start)}s')
        elif 'failed' in outcome:
            print(f'Total time: {int(time.time() - tgo)}s {file_number + 1}/{len(preprofiles)} {fileName} failed after {int(time.time() - t_start)}s')

    conn.close()
    time.sleep(2)

def execute_simulation(conn, _wait_for_real_time, sim_time, sim_time_increment, server_return_interval, data_time_step,
                       validation, reset_interval, preprodata):
    firstStep = int(round(server_return_interval / data_time_step)) + 1

    U = np.array((preprodata['time [s]'][0:firstStep].values,
                  preprodata['steer position cal [n.a.]'][0:firstStep].values,
                  preprodata['brake position effective [m]'][0:firstStep].values,
                  preprodata['motor torque cmd left [A_rms]'][0:firstStep].values,
                  preprodata['motor torque cmd right [A_rms]'][0:firstStep].values))
    # initial state [simulationTime, x, y, theta, vx, vy, vrot]
    X0 = [preprodata['time [s]'][0], preprodata['pose x [m]'][0], preprodata['pose y [m]'][0],
          preprodata['pose theta [rad]'][0],
          preprodata['vehicle vx [m*s^-1]'][0], preprodata['vehicle vy [m*s^-1]'][0],
          preprodata['pose vtheta [rad*s^-1]'][0]]

    ticker = threading.Event()
    i = 0
    i_max = sim_time / server_return_interval
    t_sim = time.time()
    while not ticker.wait(_wait_for_real_time):
        if i >= i_max:
            conn.send('simulation finished')
            return 'simulation finished'
        elif i > 0:
            simRange = [int(round(i * server_return_interval / data_time_step)),
                        int(round((i + 1) * server_return_interval / data_time_step)) + 1]
            U = np.vstack((preprodata['time [s]'][simRange[0]:simRange[1]].values,
                           preprodata['steer position cal [n.a.]'][simRange[0]:simRange[1]].values,
                           preprodata['brake position effective [m]'][simRange[0]:simRange[1]].values,
                           preprodata['motor torque cmd left [A_rms]'][simRange[0]:simRange[1]].values,
                           preprodata['motor torque cmd right [A_rms]'][simRange[0]:simRange[1]].values))

            if validation and i * server_return_interval % reset_interval < server_return_interval:
                currIndex = int(round(i * server_return_interval / data_time_step))
                X0[1] = preprodata['pose x [m]'][currIndex]
                X0[2] = preprodata['pose y [m]'][currIndex]
                X0[3] = preprodata['pose theta [rad]'][currIndex]
                X0[4] = preprodata['vehicle vx [m*s^-1]'][currIndex]
                X0[5] = preprodata['vehicle vy [m*s^-1]'][currIndex]
                X0[6] = preprodata['pose vtheta [rad*s^-1]'][currIndex]

        if i >= i_max-1:
            server_return_interval = round(U[0, -1] - U[0, 0], 4)
        # print('send',X0[-1])

        txt_msg = encode_request_msg_to_txt([X0, U, server_return_interval, sim_time_increment])
        conn.send(txt_msg)

        answer_msg = conn.recv()
        if 'Abort' in answer_msg:
            # print(answer_msg, end='\r')
            return 'simulation aborted'
        elif 'Kill' in answer_msg:
            return 'simulation failed'

        X1 = decode_answer_msg_from_txt(answer_msg)

        X0 = list(X1[-1, :])
        # print('receive',X0[-1])
        if i % 1 == 0.0:
            print(int(round(i / (sim_time / server_return_interval) * 100)), '% done, time: ', round(time.time() - t_sim, 1),
                  end='\r')
        i += 1


def getpreprodata(pathpreprodata):
    files = []
    for r, d, f in os.walk(pathpreprodata):
        for file in f:
            if '.pkl' in file:
                files.append(os.path.join(r, file))
    for filePath in files[0:1]:
        try:
            with open(filePath, 'rb') as f:
                preprodata = pickle.load(f)
        except:
            print('Could not open file at', filePath)
            preprodata = pd.DataFrame()

    return preprodata


if __name__ == '__main__':
    main()
