#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:25

@author: mvb
"""
import os
import numpy as np
from dataanalysisV2.dataIO import getCSV
from dataanalysisV2.mathfunction import interpolation

def setListItems(pathLogNr):
    files = []
    groups = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(pathLogNr):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

    for name in files:
        if 'pose.lidar' in name:
            groups.append(['pose x atvmu [m]', 0, 1, name, True, 0, 0, 0, 1])
            groups.append(['pose y atvmu [m]', 0, 2, name, True, 0, 0, 0, 1])
            groups.append(['pose theta [rad]', 0, 3, name, True, 5, 50, 0, 1])
            groups.append(['vehicle vx [m*s^-1]', 0, 5, name, True, 10, 100, 0, 1])
            groups.append(['vehicle vy [m*s^-1]', 0, 6, name, True, 10, 100, 0, 1])
        elif 'steer.put' in name:
            groups.append(['steer torque cmd [n.a.]', 0, 2, name, True, 0, 0, 0, 1])
        elif 'steer.get' in name:
            groups.append(['steer torque eff [n.a.]', 0, 5, name, True, 0, 0, 0, 1])
            groups.append(['steer position raw [n.a.]', 0, 8, name, True, 0, 0, 0, 1])
        elif 'status.get' in name:
            groups.append(['steer position cal [n.a.]', 0, 1, name, True, 0, 0, 0, 1])
        elif 'linmot.put' in name:
            groups.append(['brake position cmd [m]', 0, 1, name, True, 0, 0, 0, 1])
        elif 'linmot.get' in name:
            groups.append(['brake position effective [m]', 0, 1, name, True, 0, 0, 0, 1])
        elif 'rimo.put' in name:
            groups.append(['motor torque cmd left [A_rms]', 0, 1, name, True, 0, 0, 0, 1])
            groups.append(['motor torque cmd right [A_rms]', 0, 2, name, True, 0, 0, 0, 1])
        elif 'rimo.get' in name:
            groups.append(['motor rot rate left [rad*s^-1]', 0, 2, name, True, 0, 0, 0, 1])
            groups.append(['motor rot rate right [rad*s^-1]', 0, 9, name, True, 0, 0, 0, 1])
        elif 'vmu931' in name:
            groups.append(['vmu ax atvmu (forward) [m*s^-2]', 0, 2, name, True, 70, 700, 0, 1])
            groups.append(['vmu ay atvmu (left)[m*s^-2]', 0, 3, name, True, 70, 700, 0, 1])
            groups.append(['vmu vtheta [rad*s^-1]', 0, 4, name, True, 5, 50, 0, 1])

    groups.sort()
    allDataNames = []
    kartData = {}
    for name, timeIndex, dataIndex, fileName, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        try:
            dataFrame = getCSV(fileName)
            xRaw = dataFrame.iloc[:, timeIndex]
            yRaw = dataFrame.iloc[:, dataIndex]
            if name == 'pose theta [rad]':
                for i in range(len(yRaw)):
                    if yRaw[i] < -np.pi:
                        yRaw[i] = yRaw[i] + 2 * np.pi
                    if yRaw[i] > np.pi:
                        yRaw[i] = yRaw[i] - 2 * np.pi
                for i in range(len(yRaw) - 1):
                    if np.abs(yRaw[i + 1] - yRaw[i]) > 1:
                        yRaw[i + 1:] = yRaw[i + 1:] - np.sign((yRaw[i + 1] - yRaw[i])) * 2 * np.pi
            if name in ['vmu ax atvmu (forward) [m*s^-2]', 'vmu ay atvmu (left)[m*s^-2]', 'vmu vtheta [rad*s^-1]']:
                xRaw, yRaw = interpolation(xRaw, yRaw, xRaw.iloc[0], xRaw.iloc[-1], 0.001)
        except:
            print('EmptyDataError for ', name, ': could not read data from file ', fileName)
            xRaw, yRaw = [0], [0]

        kartData[name] = {}
        kartData[name]['data'] = [list(xRaw), list(yRaw)]  # item.data = [x_data, y_data]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    if len(groups) == 18:
        pass
#        print('Data status: complete')
    else:
        print('ACHTUNG! Missing Data in ', pathLogNr)

    # Add Preprocessed Data
    groups = []
    groups.append(['pose x [m]', ['pose x atvmu'], True, 5, 50, 1, 1])
    groups.append(['pose y [m]', ['pose y atvmu'], True, 5, 50, 1, 1])
    groups.append(['pose vx [m*s^-1]', ['pose x'], True, 5, 50, 1, 1])
    groups.append(['pose vy [m*s^-1]', ['pose y'], True, 5, 50, 1, 1])
    groups.append(['pose vtheta [rad*s^-1]', ['pose theta'], True, 5, 50, 1, 1])
    groups.append(['vehicle ax local [m*s^-2]', ['vehicle vx'], True, 0, 0, 1, 1])
    groups.append(['vehicle ay local [m*s^-2]', ['vehicle vy'], True, 0, 0, 1, 1])
    groups.append(['pose ax [m*s^-2]', ['pose vx'], True, 20, 200, 2, 1])
    groups.append(['pose ay [m*s^-2]', ['pose vy'], True, 20, 200, 2, 1])
    groups.append(['pose atheta [rad*s^-2]', ['pose vtheta'], True, 0, 0, 2, 1])
    groups.append(['vehicle slip angle [rad]', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 2, 1])
    groups.append(['vmu ax [m*s^-2]', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vmu ay [m*s^-2]', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vehicle ax total [m*s^-2]',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle','vehicle vx', 'vehicle vy'],
                   True, 0, 0, 3, 1])
    groups.append(['vehicle ay total [m*s^-2]',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'vehicle vy'],
                   True, 0, 0, 3, 1])
    groups.append(['vehicle ax only transl [m*s^-2]',
             ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0, 0, 3, 1])
    groups.append(['vehicle ay only transl [m*s^-2]',
             ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0, 0, 3, 1])
    groups.append(['MH power accel rimo left [m*s^-2]',
                   ['motor torque cmd left', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx'],
                   True, 0, 0, 4, 1])
    groups.append(['MH power accel rimo right [m*s^-2]',
                   ['motor torque cmd right', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx'],
                   True, 0, 0, 4, 1])
    groups.append(['MH AB [m*s^-2]',
                   ['pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'MH power accel rimo left', 'MH power accel rimo right'],
                   True, 0, 0, 5, 1])
    groups.append(['MH TV [rad*s^-2]',
                   ['pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'MH power accel rimo left', 'MH power accel rimo right'],
                   True, 0, 0, 5, 1])
    groups.append(['MH BETA [rad]',
                   ['steer position cal'],
                   True, 0, 0, 1, 1])

    for name, dep, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        kartData[name] = {}
        kartData[name]['data'] = [[], []]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    return kartData, allDataNames