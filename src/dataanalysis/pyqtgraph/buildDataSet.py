#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import dataIO as dio
import preprocess as prep

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from collections import defaultdict
import time


def main():
    t = time.time()

    pathRootData = '/home/mvb/0_ETH/MasterThesis/Logs_GoKart/LogData/dynamics'
    testDays = dio.getDirectories(pathRootData)
    testDays.sort()
    pathTestDay = ''
    pathLogNr = ''

    # print(testDays)
    # print(logNrs)
    j = 0
    i = 0
    for day in testDays:
        pathTestDay = pathRootData + '/' + day
        logNrs = dio.getDirectories(pathTestDay)
        logNrs.sort()
        j += 1

        for log in logNrs:
            print(log, j, i)
            i += 1
            pathLogNr = pathTestDay + '/' + log

    pathTestDay = pathRootData + '/' + testDays[-1]
    logNrs = dio.getDirectories(pathTestDay)
    logNrs.sort()
    pathLogNr = pathTestDay + '/' + logNrs[21]

    kartData = setListItems(pathLogNr)
    printTree(kartData)
    print(time.time() - t)


def printTree(kartData):
    for key, value in kartData.items():
        print(key, len(kartData[key]['data'][1]), len(kartData[key]['info']))


#        for keyy, valuee in kartData[key].items() :
#            print('   ' + keyy + str(len(valuee)))

def setListItems(pathLogNr):
    files = []
    groups = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(pathLogNr):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

    for name in files:
        if 'pose.smooth' in name:
            groups.append(['pose x', 0, 1, name, True, 0.1, 1, 0, 1])
            groups.append(['pose y', 0, 2, name, True, 0.1, 1, 0, 1])
            groups.append(['pose theta', 0, 3, name, True, 0.1, 1, 0, 1])
        elif 'steer.put' in name:
            groups.append(['steer torque cmd', 0, 2, name, True, 0.1, 1, 0, 1])
        elif 'steer.get' in name:
            groups.append(['steer torque eff', 0, 5, name, True, 0.1, 1, 0, 1])
            groups.append(['steer position raw', 0, 8, name, True, 0.1, 1, 0, 1])
        elif 'status.get' in name:
            groups.append(['steer position cal', 0, 1, name, True, 0.1, 1, 0, 1])
        elif 'linmot.put' in name:
            groups.append(['brake position cmd', 0, 1, name, True, 0.1, 1, 0, 1])
        elif 'linmot.get' in name:
            groups.append(['brake position effective', 0, 1, name, True, 0.1, 1, 0, 1])
        elif 'rimo.put' in name:
            groups.append(['motor torque cmd left', 0, 1, name, True, 0.1, 1, 0, 1])
            groups.append(['motor torque cmd right', 0, 2, name, True, 0.1, 1, 0, 1])
        elif 'rimo.get' in name:
            groups.append(['motor rot rate left', 0, 2, name, True, 0.1, 1, 0, 1])
            groups.append(['motor rot rate right', 0, 9, name, True, 0.1, 1, 0, 1])
        elif 'vmu931' in name:
            groups.append(['accel x (forward)', 0, 2, name, True, 70, 700, 0, 1])
            groups.append(['accel y (left)', 0, 3, name, True, 70, 700, 0, 1])
            groups.append(['accel theta', 0, 4, name, True, 5, 50, 0, 1])

    groups.sort()
    allDataNames = []
    kartData = defaultdict(dict)
    for name, timeIndex, dataIndex, fileName, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        dataFrame = dio.getCSV(fileName)
        xRaw = dataFrame.iloc[:, timeIndex]
        yRaw = dataFrame.iloc[:, dataIndex]
        if name == 'pose theta':
            for i in range(len(yRaw)):
                if yRaw[i] < -np.pi:
                    yRaw[i] = yRaw[i] + 2 * np.pi
                if yRaw[i] > np.pi:
                    yRaw[i] = yRaw[i] - 2 * np.pi
        if name in ['accel x (forward)', 'accel y (left)', 'accel theta']:
            xRaw, yRaw = prep.interpolation(xRaw, yRaw, 0.001)
        kartData[name]['data'] = [list(xRaw), list(yRaw)]  # item.data = [x_data, y_data]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    if len(groups) == 16:
        print('Data status: complete')
    else:
        print('ACHTUNG! Missing Data!')

    # Add Preprocessed Data
    groups = []
    groups.append(['pose vx', ['pose x'], True, 1, 10, 1, 1])
    groups.append(['pose vy', ['pose y'], True, 1, 10, 1, 1])
    groups.append(['pose vtheta', ['pose theta'], True, 0.1, 1, 1, 1])
    groups.append(['pose ax', ['pose vx'], True, 0.1, 1, 2, 1])
    groups.append(['pose ay', ['pose vy'], True, 0.1, 1, 2, 1])
    groups.append(['pose atheta', ['pose vtheta'], True, 0.1, 1, 2, 1])
    groups.append(['vehicle slip angle', ['pose theta', 'pose vx', 'pose vy'], True, 0.1, 1, 2, 1])
    groups.append(['vehicle vx', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0.1, 1, 3, 1])
    groups.append(['vehicle vy', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0.1, 1, 3, 1])
    groups.append(['vehicle ax total',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle',
                    'vehicle vx', 'vehicle vy'], True, 5, 50, 3, 1])
    groups.append(['vehicle ay total',
                   ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle',
                    'vehicle vx', 'vehicle vy'], True, 0.1, 1, 3, 1])
    groups.append(
            ['vehicle ax only transl', ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0.1, 1, 3, 1])
    groups.append(
            ['vehicle ay only transl', ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'],
             True, 0.1, 1, 3, 1])

    for name, dep, vis, sig, wid, order, scale in groups:
        allDataNames.append(name)
        kartData[name]['data'] = [[], []]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    kartData = updateData(kartData, allDataNames)
    return kartData


def updateData(kartData, dataNames):
    print('dataNames', dataNames)
    for name in dataNames:
        #        printTree(kartData)
        kartData = preProcessing(kartData, name)
        sigma = kartData[name]['info'][1]
        width = kartData[name]['info'][2]
        yOld = kartData[name]['data'][1]
        trunc = (((width - 1) / 2) - 0.5) / sigma
        yNew = gaussian_filter1d(yOld, sigma, truncate=trunc)
        kartData[name]['data'][1] = yNew
    return kartData


def preProcessing(kartData, name):
    differentiate = 'pose vx', 'pose vy', 'pose vtheta', 'pose ax', 'pose ay', 'pose atheta'
    differentiateFrom = 'pose x', 'pose y', 'pose theta', 'pose vx', 'pose vy', 'pose vtheta'
    if name in differentiate:
        index = differentiate.index(name)
        nameFrom = differentiateFrom[index]
        t, dydt = prep.derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                       kartData[nameFrom]['data'][1])
        kartData[name]['data'] = [list(t), list(dydt)]


    elif name == 'vehicle slip angle':
        x = kartData['pose vx']['data'][0]
        vx = kartData['pose vx']['data'][1]
        theta = kartData['pose theta']['data'][1]
        vy = kartData['pose vy']['data'][1]
        y = theta[:-1] - np.arctan2(vy, vx)

        for i in range(len(y)):
            if y[i] < -np.pi:
                y[i] = y[i] + 2 * np.pi
            if y[i] > np.pi:
                y[i] = y[i] - 2 * np.pi

        kartData[name]['data'] = [x, y]

    elif name in ['vehicle vx', 'vehicle vy']:
        x = kartData['pose vx']['data'][0]
        vx = kartData['pose vx']['data'][1]
        vy = kartData['pose vy']['data'][1]
        slipAngle = kartData['vehicle slip angle']['data'][1]

        if name == 'vehicle vx':
            y = np.sqrt(vx ** 2 + vy ** 2) * np.cos(slipAngle)
        else:
            y = np.sqrt(vx ** 2 + vy ** 2) * np.sin(slipAngle)

        kartData[name]['data'] = [x, y]

    elif name in ['vehicle ax total', 'vehicle ay total']:
        x = kartData['vehicle vx']['data'][0]
        vx = kartData['vehicle vx']['data'][1]
        vy = kartData['vehicle vy']['data'][1]
        vtheta = kartData['pose vtheta']['data'][1]
        if name == 'vehicle ax total':
            nameFrom = 'vehicle vx'
            t, dydt = prep.derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                           kartData[nameFrom]['data'][1])
            y = dydt - (vtheta * vy)[:-1]
        else:
            nameFrom = 'vehicle vy'
            t, dydt = prep.derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                           kartData[nameFrom]['data'][1])
            y = dydt + (vtheta * vx)[:-1]

        kartData[name]['data'] = [x[:-1], y]

    elif name in ['vehicle ax only transl', 'vehicle ay only transl']:
        x = kartData['pose ax']['data'][0]
        ax = kartData['pose ax']['data'][1]
        ay = kartData['pose ay']['data'][1]
        theta = kartData['pose theta']['data'][1]
        if name == 'vehicle ax only transl':
            y = ax * np.cos(theta[:-2]) + ay * np.sin(theta[:-2])
        else:
            y = ay * np.cos(theta[:-2]) - ax * np.sin(theta[:-2])

        kartData[name]['data'] = [x, y]
    return kartData


if __name__ == '__main__':
    main()
