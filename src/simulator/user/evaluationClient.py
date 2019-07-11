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
    #___user inputs
    pathsavedata = sys.argv[1]
    pathpreprodata = sys.argv[2]
    preprofiles = sys.argv[3:]
    # print(preprofiles)

    # preprofiles = preprofiles.split(',')[:-1]
    # pathpreprodata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/_33333333333333' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # preprofiles = ['20190521T101604_03_sampledlogdata.pkl']

    validation = True
    validationhorizon = 1      #[s] time inteval after which initial conditions are reset to values from log data

    real_time = True
    server_return_interval = 1  # [s] simulation time after which result is returned from server
    real_time_factor = 1
    _wait_for_real_time = 0

    if real_time:
        server_return_interval = 0.1*real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval*0.9*(1.0/real_time_factor)

    # preprodata = getpreprodata(pathpreprodata)

    connected = False
    address = ('localhost', 6000)
    while not connected:
        try:
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            # print('ConnectionRefusedError')
            pass

    for fileName in preprofiles:
        filePath = pathpreprodata + '/' + fileName
        try:
            with open(filePath, 'rb') as f:
                preprodata = pickle.load(f)
        except:
            print('Could not open file at', filePath)
            preprodata = pd.DataFrame()
            raise

        #___simulation parameters
        data_time_step = np.round(preprodata['time [s]'].iloc[1] - preprodata['time [s]'].iloc[0],3)  # [s] Log data sampling time step
        sim_time_increment = data_time_step     # [s] time increment used in integration scheme inside simulation server (time step for logged simulation data)
        simTime = preprodata['time [s]'].iloc[-1]#/7.0*4.0  # [s] Total simulation time

        # initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
        # X0 = [preprodata['time [s]'][0], preprodata['pose x [m]'][0], preprodata['pose y [m]'][0], preprodata['pose theta [rad]'][0],
        #       preprodata['vehicle vx [m*s^-1]'][0], preprodata['vehicle vy [m*s^-1]'][0], preprodata['pose vtheta [rad*s^-1]'][0]]

        X0 = [preprodata['time [s]'][0], preprodata['pose x [m]'][0], preprodata['pose y [m]'][0],
              preprodata['pose theta [rad]'][0],
              preprodata['vehicle vx [m*s^-1]'][0], preprodata['vehicle vy [m*s^-1]'][0],
              preprodata['pose vtheta [rad*s^-1]'][0]]

        # ______^^^______

        runSimulation = True
        ttot = time.time()
        # while runSimulation:
        print('Simulation with file ', fileName)
        firstStep = int(round(server_return_interval/data_time_step))+1

        U = np.array((preprodata['time [s]'][0:firstStep].values,
                      preprodata['steer position cal [n.a.]'][0:firstStep].values,
                      preprodata['brake position effective [m]'][0:firstStep].values,
                      preprodata['motor torque cmd left [A_rms]'][0:firstStep].values,
                      preprodata['motor torque cmd right [A_rms]'][0:firstStep].values))

        # for i in range(0,int(simTime/server_return_interval)):
        ticker = threading.Event()
        i = 0
        tgo = time.time()

        while not ticker.wait(_wait_for_real_time):
            if i >= int(simTime/server_return_interval):
                conn.send('simulation finished')
                break
            elif i > 0:
                simRange = [int(round(i * server_return_interval/data_time_step)), int(round((i+1) * server_return_interval/data_time_step))+1]

                U = np.vstack((preprodata['time [s]'][simRange[0]:simRange[1]].values,
                               preprodata['steer position cal [n.a.]'][simRange[0]:simRange[1]].values,
                               preprodata['brake position effective [m]'][simRange[0]:simRange[1]].values,
                               preprodata['motor torque cmd left [A_rms]'][simRange[0]:simRange[1]].values,
                               preprodata['motor torque cmd right [A_rms]'][simRange[0]:simRange[1]].values))

                if validation and i*server_return_interval % validationhorizon < server_return_interval:
                    currIndex = int(round(i * server_return_interval / data_time_step))
                    X0[1] = preprodata['pose x [m]'][currIndex]
                    X0[2] = preprodata['pose y [m]'][currIndex]
                    X0[3] = preprodata['pose theta [rad]'][currIndex]
                    X0[4] = preprodata['vehicle vx [m*s^-1]'][currIndex]
                    X0[5] = preprodata['vehicle vy [m*s^-1]'][currIndex]
                    X0[6] = preprodata['pose vtheta [rad*s^-1]'][currIndex]
            # else:
            #     tgo = time.time()

            txt_msg = encode_request_msg_to_txt([X0, U, server_return_interval, sim_time_increment])

            # conn.send([X0, U, server_return_interval, sim_time_increment])
            conn.send(txt_msg)

            answer_msg = conn.recv()
            X1 = decode_answer_msg_from_txt(answer_msg)

            X0 = list(X1[-1,:])
            if i%10 == 0.0:
                print(int(round(i/(simTime/server_return_interval)*100)), '% done, time: ', time.time()-tgo, end='\r')
            i += 1

        print('Success! Time overall: ', time.time()-tgo)

        # generate simulation info file and store it in target folder
        try:
            _ = open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'r')
            os.remove(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt')
            with open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'a') as the_file:
                if validation:
                    the_file.write('simulation mode:                    evaluation' + '\n')
                    the_file.write('initial condition reset interval:   ' + str(validationhorizon) + 's\n')
                else:
                    the_file.write('simulation mode:                    normal simulation' + '\n')
                the_file.write('source folder:                      ' + pathpreprodata + '\n')
                the_file.write('simulation time step:               ' + str(server_return_interval) + 's\n')
                the_file.write('total simulation time:              ' + str(simTime) + 's\n')
                the_file.write('time step in data:                  ' + str(data_time_step) + 's\n')
                the_file.write('initial conditions:                 ' + str(X0[0]) + '\n')
                for item in X0[1:]:
                    the_file.write('                                    ' + str(item) + '\n')
        except FileNotFoundError:
            with open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'a') as the_file:
                if validation:
                    the_file.write('simulation mode:                    evaluation' + '\n')
                    the_file.write('initial condition reset interval:   ' + str(validationhorizon) + 's\n')
                else:
                    the_file.write('simulation mode:                    normal simulation' + '\n')
                the_file.write('source folder:                      ' + pathpreprodata + '\n')
                the_file.write('simulation time step:               ' + str(server_return_interval) + 's\n')
                the_file.write('total simulation time:              ' + str(simTime) + 's\n')
                the_file.write('time step in data:                  ' + str(data_time_step) + 's\n')
                the_file.write('initial conditions:                 ' + str(X0[0]) + '\n')
                for item in X0[1:]:
                    the_file.write('                                    ' + str(item) + '\n')
    # print('connection closed')
    conn.close()
    time.sleep(2)
    # print('Creating reference signal for evaluation...')
    # evalRef.main()
    # print('Evaluating results...')
    # evalCalc.main()
    # print('Evaluation complete!')



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