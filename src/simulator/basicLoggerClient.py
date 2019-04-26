#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:40:03 2019

@author: mvb
"""
from multiprocessing.connection import Client
import numpy as np
import pandas as pd
import datetime
import os
import sys

def main():
    savePath = sys.argv[1]
    fileNames = sys.argv[2:]
    # fileNames = fileNames.split(',')[:-1]
    fileNameIndex = 0

    connected = False
    while not connected:
        try:
            logAddress = ('localhost', 6002)  # family is deduced to be 'AF_INET'
            logConn = Client(logAddress, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            pass


    while True:
        try:
            savePathName = savePath + '/' + fileNames[fileNameIndex][:-19] + '_simulationlog.csv'
        except IndexError:
            print('No more files to read. Closing logger.')
        logClient(savePathName, logConn)
        fileNameIndex += 1

def logClient(savePathName, logConn):

    Xall = np.array([[]])
    
    runLogger = True
    while runLogger:
        msg = logConn.recv()
        if len(msg) == 2:
            X = msg[0]
            U = msg[1]
            XU = np.concatenate((X, np.transpose(U)), axis=1)
            print(X)
            print(U)
        else:
            XU = msg


        if XU[0,0] != 'finished':
            if Xall.shape[1] < 1:
                Xall = XU
            else:
                Xall = Xall[:-1]
                Xall = np.concatenate((Xall,XU))
        else:
            dataFrame = pd.DataFrame(data=Xall)  # 1st row as the column names
            dataFrame.to_csv(savePathName, index = False, header = ['time', 'pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'MH BETA', 'MH AB', 'MH TV'])
            print('Simulation data saved to ', savePathName)
            runLogger = False

            
if __name__ == '__main__':
    main()