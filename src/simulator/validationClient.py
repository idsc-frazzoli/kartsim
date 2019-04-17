#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import time
import os
import pickle
import pandas as pd
import sys

import dataanalysis.pyqtgraph.dataIO as dIO

def main():
    #___user inputs
    #initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]

    # pathrootpreprodata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'
    # preprofolders = dIO.getDirectories(pathrootpreprodata)
    # preprofolders.sort()
    # defaultprepro = preprofolders[-1]
    # pathpreprodata = pathrootpreprodata + '/' + defaultprepro
    pathsavedata = sys.argv[1]
    pathpreprodata = sys.argv[2]
    # pathpreprodata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/20190411-135142_MarcsModel' #path where all the raw, sorted data is that you want to sample and or batch and or split

    validation = True
    validationhorizon = 1      #[s] time horizon for which simulation runs until initial conditions are reset to values from log data

    preprodata = getpreprodata(pathpreprodata)
    
    #simulation parameters
    dataStep = preprodata['time'].iloc[1]-preprodata['time'].iloc[0]       #[s] Log data sampling time step
    simStep = 0.01                                                   #[s] Simulation time step
    simTime = preprodata['time'].iloc[-1]                              #[s] Total simulation time
    print(dataStep, simStep, simTime, int(simTime/simStep))
    #initial state
    X0 = [preprodata['time'][0] , preprodata['pose x'][0], preprodata['pose y'][0], preprodata['pose theta'][0], preprodata['vehicle vx'][0], preprodata['vehicle vy'][0], preprodata['pose vtheta'][0], preprodata['MH BETA'][0], preprodata['MH AB'][0], preprodata['MH TV'][0]]
    # X0 = [preprodata['time'][1000] , preprodata['pose x'][1000], preprodata['pose y'][1000], preprodata['pose theta'][1000], preprodata['vehicle vx'][1000], preprodata['vehicle vy'][1000], preprodata['pose vtheta'][1000], preprodata['MH BETA'][1000], preprodata['MH AB'][1000], preprodata['MH TV'][1000]]
    # ______^^^______

    #generate simulation info file and store it in target folder
    try:
        _ = open(pathsavedata + '/simulationinfo.txt', 'r')
        os.remove(pathsavedata + '/simulationinfo.txt')
        with open(pathsavedata + '/simulationinfo.txt', 'a') as the_file:
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
        print('Simulation info saved to file: ', pathsavedata + '/simulationinfo.txt')
    except FileNotFoundError:
        with open(pathsavedata + '/simulationinfo.txt', 'a') as the_file:
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
        print('Simulation info saved to file', pathsavedata + '/simulationinfo.txt')
    
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'kartSim2019')
    runSimulation = True
    while runSimulation:
#        ['pose x','pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'vmu ax (forward)', 
#                    'vmu ay (left)', 'pose atheta', 'MH AB', 'MH TV', 'MH BETA', ]
        for i in range(0,int(simTime/simStep)):
            if i > 0:
                currIndex = int(round(i * simStep/dataStep))
                X0[7] = preprodata['MH BETA'][currIndex]
                X0[8] = preprodata['MH AB'][currIndex]
                X0[9] = preprodata['MH TV'][currIndex]
                if validation and i*simStep % validationhorizon < simStep:
                    X0[1] = preprodata['pose x'][currIndex]
                    X0[2] = preprodata['pose y'][currIndex]
                    X0[3] = preprodata['pose theta'][currIndex]
                    X0[4] = preprodata['vehicle vx'][currIndex]
                    X0[5] = preprodata['vehicle vy'][currIndex]
                    X0[6] = preprodata['pose vtheta'][currIndex]
            else:
                tgo = time.time()
                tint = tgo
            conn.send([X0,simStep])
            
            X1 = conn.recv()
                
            X0 = list(X1[-1,:])
            
            if i%int(simTime/simStep/20) == 0.0:
                print(int(i/(simTime/simStep)*100), '% done, time: ', time.time()-tint)
                tint = time.time()
#            time.sleep(1)
        runSimulation = False
        print('time overall: ', time.time()-tgo)




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
            print('Parameter file for preprocessing located and opened.')
        except:
            print('Parameter file for preprocessing does not exist. Creating file...')
            preprodata = pd.DataFrame()

    return preprodata

if __name__ == '__main__':
    main()