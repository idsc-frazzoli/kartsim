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


def main():
    savePath = '/home/mvb/0_ETH/01_MasterThesis/SimData'
    fileName = 'KartsimLog.csv'
    
    
    
    

    logAddress = ('localhost', 6002)     # family is deduced to be 'AF_INET'
    logConn = Client(logAddress, authkey=b'kartSim2019')
    Xall = np.array([[]])
    
    runLogger = True
    newLog = True
    while runLogger:
        X1 = logConn.recv()
        if newLog:
            currentDT = datetime.datetime.now()
            folderName = currentDT.strftime("%Y%m%d-%H%M%S")
            folderPath = savePath + '/' + folderName
            filePathName = folderPath + '/' + fileName
            try:
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
            except OSError:
                print('Error: Creating directory: ', folderPath)
            newLog = False
#        tgo = time.time()
        if X1[0,0] != 'finished':
            if Xall.shape[0] < 2:
                Xall = X1
            else:
                Xall = np.concatenate((Xall,X1[1:,:]))
        else:
            print('kill logger')
            print('Xall: ',Xall[:30,0])
            dataFrame = pd.DataFrame(data=Xall[1:,:],    # values
                                     columns=Xall[0,:])  # 1st row as the column names
            print(dataFrame)
            dataFrame.to_csv(filePathName, index = False, header = ['time', 'x', 'y', 'theta', 'vx', 'vy', 'vrot', 'steerangle', 'accRearAxle', 'tv'])
            print('Simulation data saved to ', filePathName)
            Xall = np.array([[]])
            newLog = True

#            pd.DataFrame.to_csv(dataFrame,)
            
            
if __name__ == '__main__':
    main()