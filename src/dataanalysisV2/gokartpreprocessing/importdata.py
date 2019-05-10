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
            groups.append(['pose x atvmu', 0, 1, name, True, 0, 0, 0, 1])
            groups.append(['pose y atvmu', 0, 2, name, True, 0, 0, 0, 1])
            groups.append(['pose theta', 0, 3, name, True, 5, 50, 0, 1])
            groups.append(['vehicle vx', 0, 5, name, True, 10, 100, 0, 1])
            groups.append(['vehicle vy', 0, 6, name, True, 10, 100, 0, 1])
        elif 'steer.put' in name:
            groups.append(['steer torque cmd', 0, 2, name, True, 0, 0, 0, 1])
        elif 'steer.get' in name:
            groups.append(['steer torque eff', 0, 5, name, True, 0, 0, 0, 1])
            groups.append(['steer position raw', 0, 8, name, True, 0, 0, 0, 1])
        elif 'status.get' in name:
            groups.append(['steer position cal', 0, 1, name, True, 0, 0, 0, 1])
        elif 'linmot.put' in name:
            groups.append(['brake position cmd', 0, 1, name, True, 0, 0, 0, 1])
        elif 'linmot.get' in name:
            groups.append(['brake position effective', 0, 1, name, True, 0, 0, 0, 1])
        elif 'rimo.put' in name:
            groups.append(['motor torque cmd left', 0, 1, name, True, 0, 0, 0, 1])
            groups.append(['motor torque cmd right', 0, 2, name, True, 0, 0, 0, 1])
        elif 'rimo.get' in name:
            groups.append(['motor rot rate left', 0, 2, name, True, 0, 0, 0, 1])
            groups.append(['motor rot rate right', 0, 9, name, True, 0, 0, 0, 1])
        elif 'vmu931' in name:
            groups.append(['vmu ax atvmu (forward)', 0, 2, name, True, 70, 700, 0, 1])
            groups.append(['vmu ay atvmu (left)', 0, 3, name, True, 70, 700, 0, 1])
            groups.append(['vmu vtheta', 0, 4, name, True, 5, 50, 0, 1])

    groups.sort()
    allDataNames = []
    kartData = {}
    for name, timeIndex, dataIndex, fileName, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        try:
            dataFrame = getCSV(fileName)
            xRaw = dataFrame.iloc[:, timeIndex]
            yRaw = dataFrame.iloc[:, dataIndex]

            if name == 'pose theta':
                for i in range(len(yRaw)):
                    if yRaw[i] < -np.pi:
                        yRaw[i] = yRaw[i] + 2 * np.pi
                    if yRaw[i] > np.pi:
                        yRaw[i] = yRaw[i] - 2 * np.pi
                for i in range(len(yRaw) - 1):
                    if np.abs(yRaw[i + 1] - yRaw[i]) > 1:
                        yRaw[i + 1:] = yRaw[i + 1:] - np.sign((yRaw[i + 1] - yRaw[i])) * 2 * np.pi
            if name in ['vmu ax atvmu (forward)', 'vmu ay atvmu (left)', 'vmu vtheta']:
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
    groups.append(['pose x', ['pose x atvmu'], True, 5, 50, 1, 1])
    groups.append(['pose y', ['pose y atvmu'], True, 5, 50, 1, 1])
    groups.append(['pose vx', ['pose x'], True, 5, 50, 1, 1])
    groups.append(['pose vy', ['pose y'], True, 5, 50, 1, 1])
    groups.append(['pose vtheta', ['pose theta'], True, 5, 50, 1, 1])
    groups.append(['vehicle ax local', ['vehicle vx'], True, 0, 0, 1, 1])
    groups.append(['vehicle ay local', ['vehicle vy'], True, 0, 0, 1, 1])
    groups.append(['pose ax', ['pose vx'], True, 20, 200, 2, 1])
    groups.append(['pose ay', ['pose vy'], True, 20, 200, 2, 1])
    groups.append(['pose atheta', ['pose vtheta'], True, 0, 0, 2, 1])
    groups.append(['vehicle slip angle', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 2, 1])
    groups.append(['vmu ax', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vmu ay', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vehicle ax total',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle','vehicle vx', 'vehicle vy'],
                   True, 0, 0, 3, 1])
    groups.append(['vehicle ay total',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'vehicle vy'],
                   True, 0, 0, 3, 1])
    groups.append(['vehicle ax only transl',
             ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0, 0, 3, 1])
    groups.append(['vehicle ay only transl',
             ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0, 0, 3, 1])
    groups.append(['MH power accel rimo left',
                   ['motor torque cmd left', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx'],
                   True, 0, 0, 4, 1])
    groups.append(['MH power accel rimo right',
                   ['motor torque cmd right', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx'],
                   True, 0, 0, 4, 1])
    groups.append(['MH AB',
                   ['pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'MH power accel rimo left', 'MH power accel rimo right'],
                   True, 0, 0, 5, 1])
    groups.append(['MH TV',
                   ['pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'MH power accel rimo left', 'MH power accel rimo right'],
                   True, 0, 0, 5, 1])
    groups.append(['MH BETA',
                   ['steer position cal'],
                   True, 0, 0, 1, 1])

    for name, dep, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        kartData[name] = {}
        kartData[name]['data'] = [[], []]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    return kartData, allDataNames