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

def main():
    #___user inputs
    #initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
#    X0 = [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]

    pathSimData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/20190411-135142_MarcsModel' #path where all the raw, sorted data is that you want to sample and or batch and or split

    simData = getsimdata(pathSimData)
    
    #simulation parameters
    dataStep = simData['time'].iloc[1]-simData['time'].iloc[0]       #[s] Log data sampling time step
    simStep = 0.01                                                   #[s] Simulation time step
    simTime = simData['time'].iloc[-1]                              #[s] Total simulation time
    print(dataStep, simStep, simTime, int(simTime/simStep))
    #initial state
    X0 = [simData['time'][0] , simData['pose x'][0], simData['pose y'][0], simData['pose theta'][0], simData['vehicle vx'][0], simData['vehicle vy'][0], simData['pose vtheta'][0], simData['MH BETA'][0], simData['MH AB'][0], simData['MH TV'][0]]
    # ______^^^______
    
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'kartSim2019')
    runSimulation = True
    while runSimulation:
#        ['pose x','pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'vmu ax (forward)', 
#                    'vmu ay (left)', 'pose atheta', 'MH AB', 'MH TV', 'MH BETA', ]
        for i in range(0,int(simTime/simStep)):
            if i > 0:
                currIndex = int(round(i * simStep/dataStep))
                X0[7] = simData['MH BETA'][currIndex]
                X0[8] = simData['MH AB'][currIndex]
                X0[9] = simData['MH TV'][currIndex]
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


def getsimdata(pathSimData):
    files = []
    for r, d, f in os.walk(pathSimData):
        for file in f:
            if '.pkl' in file:
                files.append(os.path.join(r, file))
    for filePath in files[0:1]:
        try:
            with open(filePath, 'rb') as f:
                simData = pickle.load(f)
            print('Parameter file for preprocessing located and opened.')
        except:
            print('Parameter file for preprocessing does not exist. Creating file...')
            simData = pd.DataFrame()

    return simData

if __name__ == '__main__':
    main()