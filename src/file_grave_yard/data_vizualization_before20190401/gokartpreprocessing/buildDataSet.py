#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import dataIO as dio
import showrawdata.preprocess as prep

import os
import numpy as np
import pandas as pd
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d, interp2d

import time
import datetime


def main():
    t = time.time()
    
    #__User parameters
    
    pathRootData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics' #path where all the raw logfiles are

    #preprocess data and compute inferred data from raw logs
    preprocessData = True
    requiredList = ['pose x','pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta', 'vmu ax',
                    'vmu ay', 'pose atheta', 'MH AB', 'MH TV', 'MH BETA', ] #list of required raw log parameters
    saveDatasetPath = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/PreprocessedData'
    datasetTag = 'test_threeLogs'

    #check logs for missing or incomplete data
    sortOutData = False                                                         #if True: checks all the raw logfiles for missing/incomplete data
    sortOutDataOverwrite = False                                                #if True: all data in preproParams-file will be overwritten
    preproParamsFileName = 'preproParams'
    # preproParamsFilePath = pathRootData + '/' + preproParamsFileName + '.pkl'   #file where all the information about missing/incomplete data is stored
    preproParamsFilePath = pathRootData + '/' + preproParamsFileName + '.csv'   #file where all the information about missing/incomplete data is stored

    #______________^^^_________________
    
    
    try:
        preproParams = readDictFromCSV(preproParamsFilePath)
        print('Parameter file for preprocessing located and opened.')
    except:
        print('Parameter file for preprocessing does not exist. Creating file...')
        preproParams = {}
    
    if sortOutData:
        #tag logs with missing data
        preproParams = sortOut(pathRootData, preproParams, sortOutDataOverwrite)
        #save information to file
        writeDictToCSV(preproParamsFilePath, preproParams)
        print('preproParams saved to ', preproParamsFilePath)

    if preprocessData:
        kartDataAll, comp_tot = stirData(pathRootData, preproParams, requiredList)
        print('Data preprocessing completed.')
        currentDT = datetime.datetime.now()
        folderName = currentDT.strftime("%Y%m%d-%H%M%S")
        folderPath = saveDatasetPath + '/' + folderName + '_' + datasetTag
        
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error: Creating directory: ', folderPath)
        print('Now writing to file at', folderPath)
        comp_count = 0
        for key in kartDataAll:
            filePathName = folderPath + '/' + key + '.pkl'
            try:
                with open(filePathName, 'wb') as f:
                    pickle.dump(kartDataAll[key], f, pickle.HIGHEST_PROTOCOL)
                # print(key + '.pkl',' done')
                print(str(int(comp_count / comp_tot * 100)),
                      '% completed.   current file:', key + '.pkl', end='\r')
            except:
                print('Could not save ', key + '.pkl' ,' to file.')
            comp_count += 1
        
    print('Total computing time: ', time.time() - t)
      
def writeDictToCSV(filePath, dict):
    with open(filePath, 'w') as f:
        for day in sorted(dict):
            f.write("%s\n"%(day))
            for log in sorted(dict[day]):
                f.write(",%s\n" % (log))
                for topic in sorted(dict[day][log]):
                    f.write(",,%s\n" % (topic))
                    try:
                        for value in sorted(dict[day][log][topic]):
                            f.write(",,,%s\n" % (value))
                    except TypeError:
                        f.write(",,,%s\n" % (dict[day][log][topic]))

def readDictFromCSV(filePath):
    dict = {}
    with open(filePath, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.split(',')
            if line[0] != '':
                d0 = line[0]
                dict[d0] = {}
            elif line[1] != '':
                d1 = line[1]
                dict[d0][d1] = {}
            elif line[2] != '':
                d2 = line[2]
                dict[d0][d1][d2] = {}
            elif line[3] != '':
                d3 = line[3]
                dict[d0][d1][d2] = d3
    return dict
    
def stirData(pathRootData, preproParams, requiredList):
    loggoodcount = 0
    logtotcount = 0
    for day in preproParams:
        for log in preproParams[day]:
            loggoodcount += 1
            logtotcount += 1
            for topic in preproParams[day][log]:
                if topic in requiredList and int(preproParams[day][log][topic]) == 0:
                    preproParams[day][log]['goodData'] = 0
                    loggoodcount -= 1
                    break
    print(loggoodcount, 'of', logtotcount, 'logs are used for creating this dataset.')

    comp_tot = 0
    testDays = dio.getDirectories(pathRootData)
    testDays.sort()
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = dio.getDirectories(pathTestDay)
        logNrs.sort()
        for logNr in logNrs:
            if preproParams[testDay][logNr]['goodData']:
                comp_tot += 1

    kartDataAll = {}
    skipCount = 0
    comp_count = 0
    for testDay in testDays[0:1]:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = dio.getDirectories(pathTestDay)
        logNrs.sort()
    
        for logNr in logNrs:
            if preproParams[testDay][logNr]['goodData']:

                if skipCount > 0:
                    print(str(int(comp_count / comp_tot * 100)), '% completed.   current log:', logNr, '  ', skipCount, 'logs skipped', end='\r')
                    skipCount = 0
                else:
                    print(str(int(comp_count / comp_tot * 100)), '% completed.   current log:', logNr, end='\r')

                pathLogNr = pathTestDay + '/' + logNr
                kartData, allDataNames = setListItems(pathLogNr)
                kartData = updateData(kartData, allDataNames)
                
                delTopics = []
                for topic in kartData:
                    if topic not in requiredList:
                        delTopics.append(topic)
                    else:
                        kartData[topic] = kartData[topic]['data']
                for delTopic in delTopics:
                    kartData.pop(delTopic,None)
                
                kartDataAll[logNr] = kartData
                comp_count += 1

            else:
                skipCount += 1
    return kartDataAll, comp_tot


def sortOut(pathRootData, preproParams, redo):
    testDays = dio.getDirectories(pathRootData)
    testDays.sort()

    comp_tot = 0
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = dio.getDirectories(pathTestDay)
        logNrs.sort()
        for logNr in logNrs:
            comp_tot += 1

    comp_count = 0
    for testDay in testDays:
        # if testDay in preproParams and not redo:
        #    print(testDay, ' already done. Continuing with next testDay')
        #    continue
        if testDay not in preproParams:
            preproParams[testDay] = {}
        pathTestDay = pathRootData + '/' + testDay
        logNrs = dio.getDirectories(pathTestDay)
        logNrs.sort()
    
        for logNr in logNrs:
            if logNr in preproParams[testDay] and not redo:
                print(logNr, ' already done. Continuing with next logNr')
                continue
            if logNr not in preproParams[testDay]:
                preproParams[testDay][logNr] = {}
                preproParams[testDay][logNr]['goodData'] = 1
            statusInfo = logNr + ':  '
            pathLogNr = pathTestDay + '/' + logNr
            
            kartData, allDataNames = setListItems(pathLogNr)
            
            lenSteerCmd = len(kartData['steer torque cmd']['data'][1])
#            print (kartData['steer torque cmd']['data'][1][int(lenSteerCmd/10):int(lenSteerCmd/10*9)].count(0)/len(kartData['steer torque cmd']['data'][1][int(lenSteerCmd/10):int(lenSteerCmd/10*9)]))
            if kartData['steer torque cmd']['data'][1][int(lenSteerCmd/10):int(lenSteerCmd/10*9)].count(0)/len(kartData['steer torque cmd']['data'][1][int(lenSteerCmd/10):int(lenSteerCmd/10*9)]) > 0.05:
                statusInfo = statusInfo + 'steering cmd data: too many zeros...,  '
                preproParams[testDay][logNr]['steer torque cmd'] = 0
                preproParams[testDay][logNr]['MH BETA'] = 0
            elif np.abs(np.mean(kartData['steer torque cmd']['data'][1])) < 0.01 and np.std(kartData['steer torque cmd']['data'][1]) < 0.1:
                statusInfo = statusInfo + 'steering cmd data missing or insufficient,  '
                preproParams[testDay][logNr]['steer torque cmd'] = 0
                preproParams[testDay][logNr]['MH BETA'] = 0
            else:
                preproParams[testDay][logNr]['steer torque cmd'] = 1
                preproParams[testDay][logNr]['MH BETA'] = 1
                
            if np.abs(np.mean(kartData['brake position cmd']['data'][1])) < 0.005 or np.std(kartData['brake position cmd']['data'][1]) < 0.001:
                statusInfo = statusInfo + 'brake position cmd data missing or insufficient,  '
                preproParams[testDay][logNr]['brake position cmd'] = 0
                preproParams[testDay][logNr]['MH AB'] = 0
                preproParams[testDay][logNr]['MH TV'] = 0
            else:
                preproParams[testDay][logNr]['brake position cmd'] = 1
                preproParams[testDay][logNr]['MH AB'] = 1
                preproParams[testDay][logNr]['MH TV'] = 1
                
            if np.abs(np.mean(kartData['vmu ax']['data'][1])) < 0.01 and np.std(kartData['vmu ax']['data'][1]) < 0.01:
                statusInfo = statusInfo + 'vmu ax data missing or insufficient,  '
                preproParams[testDay][logNr]['vmu ax'] = 0
            else:
                preproParams[testDay][logNr]['vmu ax'] = 1
            
            if np.abs(np.mean(kartData['vmu ay']['data'][1])) < 0.01 and np.std(kartData['vmu ay']['data'][1]) < 0.05:
                statusInfo = statusInfo + 'vmu ay data missing or insufficient,  '
                preproParams[testDay][logNr]['vmu ay'] = 0
            else:
                preproParams[testDay][logNr]['vmu ay'] = 1
                
            if np.abs(np.mean(kartData['vmu vtheta']['data'][1])) < 0.01 and np.std(kartData['vmu vtheta']['data'][1]) < 0.05:
                statusInfo = statusInfo + 'vmu vtheta data missing or insufficient,  '
                preproParams[testDay][logNr]['vmu vtheta'] = 0
            else:
                preproParams[testDay][logNr]['vmu vtheta'] = 1
        
            statusInfo = statusInfo + 'done'
            comp_count += 1
            print(statusInfo)
            print(str(int(comp_count/comp_tot*100)),'% completed.  ', statusInfo, end='\r')
            
    return preproParams

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
            groups.append(['pose x atvmu', 0, 1, name, True, 0, 0, 0, 1])
            groups.append(['pose y atvmu', 0, 2, name, True, 0, 0, 0, 1])
            groups.append(['pose theta', 0, 3, name, True, 0, 0, 0, 1])
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
            dataFrame = dio.getCSV(fileName)
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
                xRaw, yRaw = prep.interpolation(xRaw, yRaw, xRaw.iloc[0], xRaw.iloc[-1], 0.001)
        except:
            print('EmptyDataError for ', name, ': could not read data from file ', fileName)
            xRaw = [0]
            yRaw = [0]
        
        kartData[name] = {}
        kartData[name]['data'] = [list(xRaw), list(yRaw)]  # item.data = [x_data, y_data]
        kartData[name]['info'] = [vis, sig, wid, order, scale]  # item.info = [visible,
        # filter_sigma, filter_width, order, scale]

    if len(groups) == 16:
        pass
#        print('Data status: complete')
    else:
        print('ACHTUNG! Missing Data in ', pathLogNr)

    # Add Preprocessed Data
    groups = []
    groups.append(['pose x', ['pose x atvmu'], True, 0, 0, 1, 1])
    groups.append(['pose y', ['pose y atvmu'], True, 0, 0, 1, 1])
    groups.append(['pose vx', ['pose x'], True, 0, 0, 1, 1])
    groups.append(['pose vy', ['pose y'], True, 0, 0, 1, 1])
    groups.append(['pose vtheta', ['pose theta'], True, 0, 0, 1, 1])
    groups.append(['pose ax', ['pose vx'], True, 0, 0, 2, 1])
    groups.append(['pose ay', ['pose vy'], True, 0, 0, 2, 1])
    groups.append(['pose atheta', ['pose vtheta'], True, 0, 0, 2, 1])
    groups.append(['vehicle slip angle', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 2, 1])
    groups.append(['vmu ax', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vmu ay', ['pose theta', 'pose vx', 'pose vy'], True, 0, 0, 3, 1])
    groups.append(['vehicle vx', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0, 0, 3, 1])
    groups.append(['vehicle vy', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0, 0, 3, 1])
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


def updateData(kartData, dataNames):
    for name in dataNames:
        kartData = preProcessing(kartData, name)
        sigma = kartData[name]['info'][1]
        width = kartData[name]['info'][2]
        yOld = kartData[name]['data'][1]
        if sigma == 0:
            yNew = yOld
        else:
            trunc = (((width - 1) / 2) - 0.5) / sigma
            yNew = gaussian_filter1d(yOld, sigma, truncate=trunc)
        kartData[name]['data'][1] = yNew
    return kartData


def preProcessing(kartData, name):
    vmu_cog = 0.48 #[m] displacement of cog to vmu wrt vmu
    
    differentiate = 'pose vx', 'pose vy', 'pose vtheta', 'pose ax', 'pose ay', 'pose atheta'
    differentiateFrom = 'pose x', 'pose y', 'pose theta', 'pose vx', 'pose vy', 'pose vtheta'
    
    if name in differentiate:
        index = differentiate.index(name)
        nameFrom = differentiateFrom[index]
        t, dydt = prep.derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                       kartData[nameFrom]['data'][1])
        kartData[name]['data'] = [list(t), list(dydt)]

    if name in ['pose x', 'pose y']:
        
        theta = kartData['pose theta']['data'][1]

        if name == 'pose x':
            x = kartData['pose x atvmu']['data'][0]
            y = kartData['pose x atvmu']['data'][1]
            y = y + vmu_cog * np.cos(theta)
        else:
            x = kartData['pose y atvmu']['data'][0]
            y = kartData['pose y atvmu']['data'][1]
            y = y + vmu_cog * np.sin(theta)

        kartData[name]['data'] = [x, y]
    
    
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
        
    elif name in ['vmu ax', 'vmu ay']:
        x = kartData['vmu ax atvmu (forward)']['data'][0]
        a = kartData['vmu ax atvmu (forward)']['data'][1]
        atheta_t = kartData['pose atheta']['data'][0]
        atheta = kartData['pose atheta']['data'][1]

        while x[0] < atheta_t[0]:
            x = np.delete(x, 0)
            a = np.delete(a, 0)
        while x[-1] > atheta_t[-1]:
            x = np.delete(x, -1)
            a = np.delete(a, -1)

        interp = interp1d(atheta_t, atheta)
        atheta = interp(x)

        if name == 'vmu ax':
            y = a
        else:
            y = a - atheta * vmu_cog

        kartData[name]['data'] = [x, y]

    elif name in ['vehicle vx', 'vehicle vy']:
        x = kartData['pose vx']['data'][0]
        vx = kartData['pose vx']['data'][1]
        vy = kartData['pose vy']['data'][1]
        slipAngle = kartData['vehicle slip angle']['data'][1]

        if name == 'vehicle vx':
            y = np.sqrt(np.square(vx) + np.square(vy)) * np.cos(slipAngle)
        else:
            y = np.sqrt(np.square(vx) + np.square(vy)) * np.sin(slipAngle)

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
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vy):
                vy = vy[:-1]

            y = dydt - (vtheta * vy)
        else:
            nameFrom = 'vehicle vy'
            t, dydt = prep.derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                           kartData[nameFrom]['data'][1])
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vx):
                vx = vx[:-1]
                
            y = dydt + (vtheta * vx)

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
        
    elif name in ['MH power accel rimo left', 'MH power accel rimo right']:
        if name == 'MH power accel rimo left':
            x = kartData['motor torque cmd left']['data'][0]
            motorPower = kartData['motor torque cmd left']['data'][1]
        else:
            x = kartData['motor torque cmd right']['data'][0]
            motorPower = kartData['motor torque cmd right']['data'][1]
        velocity_t = kartData['vehicle vx']['data'][0]
        velocity = kartData['vehicle vx']['data'][1]

        while x[0] < velocity_t[0]:
            x = np.delete(x,0)
            motorPower= np.delete(motorPower,0)
        while x[-1] > velocity_t[-1]:
            x = np.delete(x,-1)
            motorPower= np.delete(motorPower,-1)
        interp = interp1d(velocity_t, velocity)
        velocity = interp(x)

        lookupFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/data_vizualization_before20190401/showsimdata/lookup_cur_vel_to_acc.pkl'   #lookupTable file
        try:
            with open(lookupFilePath, 'rb') as f:
                lookupTable = pickle.load(f)
            # print('Lookup Table file for preprocessing located and opened.')
        except:
            print('Lookup Table file for preprocessing does not exist. Creating file...')
            lookupTable = pd.DataFrame()
        interp = interp2d(lookupTable.columns, lookupTable.index, lookupTable.values)
        powerAcceleration = [float(interp(XX,YY)) for XX,YY in zip(velocity,motorPower)]

        kartData[name]['data'] = [x, powerAcceleration]
    
    elif name == 'MH AB':
        x = kartData['MH power accel rimo left']['data'][0]
        powerAccelL = kartData['MH power accel rimo left']['data'][1]
        powerAccelR = kartData['MH power accel rimo right']['data'][1]
        brakePos_t = kartData['brake position cmd']['data'][0]
        brakePos = kartData['brake position cmd']['data'][1]
        velx_t = kartData['vehicle vx']['data'][0]
        velx = kartData['vehicle vx']['data'][1]
        
        powerAccel = np.dstack((powerAccelL,powerAccelR))
        
        AB_rimo = np.mean(powerAccel, axis=2)
        
        while x[0] < brakePos_t[0]:
            x = np.delete(x,0)
            AB_rimo = np.delete(AB_rimo,0)
        while x[-1] > brakePos_t[-1]:
            x = np.delete(x,-1)
            AB_rimo = np.delete(AB_rimo,-1)
        while x[0] < velx_t[0]:
            x = np.delete(x,0)
            AB_rimo = np.delete(AB_rimo,0)
        while x[-1] > velx_t[-1]:
            x = np.delete(x,-1)
            AB_rimo = np.delete(AB_rimo,-1)

            
        interp1B = interp1d(brakePos_t, brakePos)
        brakePos = interp1B(x)
        interp1V = interp1d(velx_t, velx)
        velx = interp1V(x)
        
        staticBrakeFunctionFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/data_vizualization_before20190401/showsimdata/staticBrakeFunction.pkl'   #static brake function file
        try:
            with open(staticBrakeFunctionFilePath, 'rb') as f:
                staticBrakeFunction = pickle.load(f)
            # print('staticBrakeFunction file for preprocessing located and opened.')
        except:
            print('staticBrakeFunction file for preprocessing does not exist. Creating file...')
            staticBrakeFunction = pd.DataFrame()
        
        interp2 = interp1d(staticBrakeFunction['brake pos'], staticBrakeFunction['deceleration'])
        deceleration = interp2(brakePos)

        for i in range(len(deceleration)):
            if velx[i] <= 0:
                deceleration[i] = 0
        
        # AB_brake = np.multiply(0.6,-deceleration)
        AB_brake = -deceleration

        AB_tot = AB_rimo + AB_brake
        
        kartData[name]['data'] = [x, AB_tot[0,:]]
            
    elif name == 'MH TV':
        x = kartData['MH power accel rimo left']['data'][0]
        powerAccelL = kartData['MH power accel rimo left']['data'][1]
        powerAccelR = kartData['MH power accel rimo right']['data'][1]
        
        TV = np.subtract(powerAccelR, powerAccelL)/2.0

        kartData[name]['data'] = [x, TV]
            
    elif name == 'MH BETA':
        x = kartData['steer position cal']['data'][0]
        steerCal = np.array(kartData['steer position cal']['data'][1])
        BETA = -0.625 * np.power(steerCal, 3) + 0.939 * steerCal
        
        kartData[name]['data'] = [x, BETA]
    return kartData


if __name__ == '__main__':
    main()
