#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import numpy as np
import time
import os
import pickle
import pandas as pd
import sys

import dataanalysis.pyqtgraph.evaluationReference as evalRef
import dataanalysis.pyqtgraph.evaluation as evalCalc

def main():
    #___user inputs

    pathsavedata = sys.argv[1]
    pathpreprodata = sys.argv[2]
    preprofiles = sys.argv[3:]

    # preprofiles = preprofiles.split(',')[:-1]
    # pathpreprodata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/20190411-135142_MarcsModel' #path where all the raw, sorted data is that you want to sample and or batch and or split

    validation = True
    validationhorizon = 1      #[s] time inteval after which initial conditions are reset to values from log data
    simStep = 0.1  # [s] Simulation time step

    preprodata = getpreprodata(pathpreprodata)

    # #simulation parameters
    # dataStep = preprodata['time'].iloc[1]-preprodata['time'].iloc[0]       #[s] Log data sampling time step
    # simStep = 0.01                                                   #[s] Simulation time step
    # simTime = preprodata['time'].iloc[-1]                              #[s] Total simulation time
    # # print(dataStep, simStep, simTime, int(simTime/simStep), len(preprodata['time']))
    #
    # #initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
    # X0 = [preprodata['time'][0] , preprodata['pose x'][0], preprodata['pose y'][0], preprodata['pose theta'][0], preprodata['vehicle vx'][0], preprodata['vehicle vy'][0], preprodata['pose vtheta'][0], preprodata['MH BETA'][0], preprodata['MH AB'][0], preprodata['MH TV'][0]]
    # # X0 = [preprodata['time'][1000] , preprodata['pose x'][1000], preprodata['pose y'][1000], preprodata['pose theta'][1000], preprodata['vehicle vx'][1000], preprodata['vehicle vy'][1000], preprodata['pose vtheta'][1000], preprodata['MH BETA'][1000], preprodata['MH AB'][1000], preprodata['MH TV'][1000]]

    # ______^^^______

    # files = []
    # for r, d, f in os.walk(pathpreprodata):
    #     for file in f:
    #         if '.pkl' in file:
    #             files.append([os.path.join(r, file), file])
    # files.sort()
    # for fileName in preprofiles:
    #     filePath = pathpreprodata + '/' + fileName
    #     try:
    #         with open(filePath, 'rb') as f:
    #             preprodata = pickle.load(f)
    #     except:
    #         print('Could not open file at', filePath)
    #         preprodata = pd.DataFrame()
    #
    #     # ___simulation parameters
    #     dataStep = preprodata['time'].iloc[1] - preprodata['time'].iloc[0]  # [s] Log data sampling time step
    #     simTime = preprodata['time'].iloc[-1]  # [s] Total simulation time
    #     X0 = [preprodata['time'][0], preprodata['pose x'][0], preprodata['pose y'][0], preprodata['pose theta'][0],
    #           preprodata['vehicle vx'][0], preprodata['vehicle vy'][0], preprodata['pose vtheta'][0],
    #           preprodata['MH BETA'][0], preprodata['MH AB'][0], preprodata['MH TV'][0]]


    # print('Simulation info saved to', pathsavedata)

    for fileName in preprofiles:
        filePath = pathpreprodata + '/' + fileName
        try:
            with open(filePath, 'rb') as f:
                preprodata = pickle.load(f)
        except:
            print('Could not open file at', filePath)
            preprodata = pd.DataFrame()

        #___simulation parameters
        dataStep = preprodata['time'].iloc[1] - preprodata['time'].iloc[0]  # [s] Log data sampling time step
        simTime = preprodata['time'].iloc[-1]  # [s] Total simulation time
        # print(dataStep, simStep, simTime, int(simTime / simStep), len(preprodata['MH BETA']))

        # initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
        X0 = [preprodata['time'][0], preprodata['pose x'][0], preprodata['pose y'][0], preprodata['pose theta'][0],
              preprodata['vehicle vx'][0], preprodata['vehicle vy'][0], preprodata['pose vtheta'][0]]
        # X0 = [preprodata['time'][0], preprodata['pose x'][0], preprodata['pose y'][0], preprodata['pose theta'][0],
        #       preprodata['vehicle vx'][0], preprodata['vehicle vy'][0], preprodata['pose vtheta'][0],
        #       preprodata['MH BETA'][0], preprodata['MH AB'][0], preprodata['MH TV'][0]]
        # X0 = [preprodata['time'][1000] , preprodata['pose x'][1000], preprodata['pose y'][1000], preprodata['pose theta'][1000], preprodata['vehicle vx'][1000], preprodata['vehicle vy'][1000], preprodata['pose vtheta'][1000], preprodata['MH BETA'][1000], preprodata['MH AB'][1000], preprodata['MH TV'][1000]]

        # ______^^^______

        connected = False
        while not connected:
            try:
                address = ('localhost', 6000)
                conn = Client(address, authkey=b'kartSim2019')
                connected = True
            except ConnectionRefusedError:
                pass

        runSimulation = True
        while runSimulation:
    #        ['pose x','pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'vmu ax (forward)',
    #                    'vmu ay (left)', 'pose atheta', 'MH AB', 'MH TV', 'MH BETA', ]
            print('Simulating with file ', fileName)
            firstStep = int(round(simStep/dataStep))+1
            U = [preprodata['time'][0:firstStep].values,
                 preprodata['MH BETA'][0:firstStep].values,
                 preprodata['MH AB'][0:firstStep].values,
                 preprodata['MH TV'][0:firstStep].values]
            for i in range(0,int(simTime/simStep)):
                if i > 0:
                    # currIndex = int(round(i * simStep/dataStep))
                    # X0[7] = preprodata['MH BETA'][currIndex]
                    # X0[8] = preprodata['MH AB'][currIndex]
                    # X0[9] = preprodata['MH TV'][currIndex]

                    simRange = [int(round(i * simStep/dataStep)), int(round((i+1) * simStep/dataStep))+1]

                    # U = [preprodata['time'][simRange[0]:simRange[1]].values,
                    #        preprodata['MH BETA'][simRange[0]:simRange[1]].values,
                    #        preprodata['MH AB'][simRange[0]:simRange[1]].values,
                    #        preprodata['MH TV'][simRange[0]:simRange[1]].values]
                    U = np.vstack((preprodata['time'][simRange[0]:simRange[1]].values,
                                     preprodata['MH BETA'][simRange[0]:simRange[1]].values,
                                     preprodata['MH AB'][simRange[0]:simRange[1]].values,
                                     preprodata['MH TV'][simRange[0]:simRange[1]].values))

                    if validation and i*simStep % validationhorizon < simStep:
                        currIndex = int(round(i * simStep / dataStep))
                        X0[1] = preprodata['pose x'][currIndex]
                        X0[2] = preprodata['pose y'][currIndex]
                        X0[3] = preprodata['pose theta'][currIndex]
                        X0[4] = preprodata['vehicle vx'][currIndex]
                        X0[5] = preprodata['vehicle vy'][currIndex]
                        X0[6] = preprodata['pose vtheta'][currIndex]
                else:
                    tgo = time.time()

                conn.send([X0,U,simStep])

                X1 = conn.recv()

                X0 = list(X1[-1,:])
                if i%int(simTime/simStep/20) == 0.0:
                    print(int(round(i/(simTime/simStep)*100)), '% done, time: ', time.time()-tgo, end='\r')
    #            time.sleep(1)
            runSimulation = False
            conn.close()
            print('time overall: ', time.time()-tgo)

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
                the_file.write('simulation time step:               ' + str(simStep) + 's\n')
                the_file.write('total simulation time:              ' + str(simTime) + 's\n')
                the_file.write('time step in data:                  ' + str(dataStep) + 's\n')
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
                the_file.write('simulation time step:               ' + str(simStep) + 's\n')
                the_file.write('total simulation time:              ' + str(simTime) + 's\n')
                the_file.write('time step in data:                  ' + str(dataStep) + 's\n')
                the_file.write('initial conditions:                 ' + str(X0[0]) + '\n')
                for item in X0[1:]:
                    the_file.write('                                    ' + str(item) + '\n')

    time.sleep(2)
    print('Creating reference signal for evaluation...')
    evalRef.main()
    print('Evaluating results...')
    evalCalc.main()



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