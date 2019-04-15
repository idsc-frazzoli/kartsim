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
    fileName = 'kartsimlog.csv'
    # simTag = 'validation'
    simTag = sys.argv[2]

    # currentDT = datetime.datetime.now()
    # folderName = currentDT.strftime("%Y%m%d-%H%M%S")
    # folderPath = savePath + '/' + folderName + '_' + simTag
    savePathName = savePath + '/' + fileName
    # try:
    #     if not os.path.exists(folderPath):
    #         os.makedirs(folderPath)
    # except OSError:
    #     print('Error: Creating directory: ', folderPath)

    logAddress = ('localhost', 6002)     # family is deduced to be 'AF_INET'
    logConn = Client(logAddress, authkey=b'kartSim2019')
    Xall = np.array([[]])
    
    runLogger = True
    while runLogger:
        X1 = logConn.recv()
#        tgo = time.time()
        if X1[0,0] != 'finished':
            if Xall.shape[1] < 1:
                Xall = X1
            else:
                Xall = Xall[:-1]
                Xall = np.concatenate((Xall,X1))
        else:
            print('kill logger')

#            print('Xall: ',Xall[:30,0])
            dataFrame = pd.DataFrame(data=Xall)  # 1st row as the column names
#            print(dataFrame)
            dataFrame.to_csv(savePathName, index = False, header = ['time', 'pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'MH BETA', 'MH AB', 'MH TV'])
            print('Simulation data saved to ', savePathName)
            Xall = np.array([[]])

#            pd.DataFrame.to_csv(dataFrame,)
            
            
if __name__ == '__main__':
    main()