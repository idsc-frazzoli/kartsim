#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:40:03 2019

@author: mvb
"""
from multiprocessing.connection import Client
import numpy as np
import pandas as pd
import sys

from dataanalysisV2.mathfunction import derivative_X_dX

def main():
    savePath = sys.argv[1]
    model_type = sys.argv[2]
    model_name = sys.argv[3]
    fileNames = sys.argv[4:]

    if model_type == "mpc_dynamic":
        model_log_name = model_type
    else:
        model_log_name = model_type + '_' + model_name

    fileNameIndex = 0
    print('Simulation data will be saved to ', savePath)
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
            savePathName = savePath + '/' + fileNames[fileNameIndex][:-19] + '_{}_closedloop.csv'.format(model_log_name)
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

            X = add_acceleration(X)

            XU = np.concatenate((X, np.transpose(U)), axis=1)
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
            # dataFrame.to_csv(savePathName, index = False,
            #                  header = ['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
            #                            'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
            #                            'pose atheta [rad*s^-2]', 'MH BETA [rad]', 'MH AB [m*s^-2]', 'MH TV [rad*s^-2]',])
            dataFrame.to_csv(savePathName, index=False,
                             header=['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]',
                                     'vehicle vy [m*s^-1]',
                                     'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
                                     'pose atheta [rad*s^-2]', 'steer position cal [n.a.]',
                                     'brake position effective [m]', 'motor torque cmd left [A_rms]',
                                     'motor torque cmd right [A_rms]'])

            runLogger = False


def add_acceleration(X):
    _,ax = derivative_X_dX("", X[:, 0], X[:, 4])
    _,ay = derivative_X_dX("", X[:, 0], X[:, 5])
    _,atheta = derivative_X_dX("", X[:, 0], X[:, 6])

    ax = np.array([np.append(ax,ax[-1])])
    ay = np.array([np.append(ay,ay[-1])])
    atheta = np.array([np.append(atheta,atheta[-1])])

    # ax = np.add(ax, np.multiply(X[:,6], X[:,5]))
    # ay = np.subtract(ay, np.multiply(X[:,6], X[:,4]))

    X = np.concatenate((X, ax.transpose()), axis=1)
    X = np.concatenate((X, ay.transpose()), axis=1)
    X = np.concatenate((X, atheta.transpose()), axis=1)

    return X


if __name__ == '__main__':
    main()