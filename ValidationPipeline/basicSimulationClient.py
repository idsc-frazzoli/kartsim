#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import time

def main():
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'kartSim2019')
    
    #___user inputs
    #initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
    X0 = [0, 0, 0, 0, 0, 0, 0, 0.5, 1.8, 0.4]
    #simulation time step
    simStep = 0.1      #[s] Simulation time step
    simTime = 10        #[s] Total simulation time
    runSimulation = True
    while runSimulation:
        
        for i in range(int(simTime/simStep)):
            conn.send([X0,simStep])
            
            X1 = conn.recv()
            
            if i == 0:
                tgo = time.time()
                tint = tgo
            X0 = list(X1[-1,:])
            
            if i%int(simTime/simStep/10) == 0.0:
                print(int(i/(simTime/simStep)*100), '% done, time: ', time.time()-tint)
                tint = time.time()
#            time.sleep(1)
        runSimulation = False
        print('time overall: ', time.time()-tgo)
        
if __name__ == '__main__':
    main()