#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:37:23 2019

@author: mvb
"""
from pyqtgraph.Qt import QtCore
import numpy as np
from scipy.interpolate import interp1d, interp2d
import pandas as pd
import pickle

def interpolation(x, y, xBegin, xStop, timeStep):
    if isinstance(y,pd.Series):
        interp = interp1d(x, y)
    
        if xBegin % timeStep != 0:
            xBegin = (timeStep - xBegin % timeStep) + xBegin
        else:
            pass
    
        if xStop % timeStep != 0:
            xStop = xStop - xStop % timeStep
        else:
            pass
        
        xInterp = np.linspace(xBegin, xStop, int((xStop-xBegin)/timeStep))
        yInterp = interp(xInterp)

        return xInterp, yInterp
    
    elif isinstance(y,list):
        interp = interp1d(x, y)
    
        if xBegin % timeStep != 0:
            xBegin = (timeStep - xBegin % timeStep) + xBegin
        else:
            pass
    
        if xStop % timeStep != 0:
            xStop = xStop - xStop % timeStep
        else:
            pass
    
        xInterp = np.linspace(xBegin, xStop, int((xStop-xBegin)/timeStep))
        yInterp = interp(xInterp)

        return xInterp, yInterp


def preProcessing(self, name):
    availableDataList = [item[0] for item in self.availableData]
    vmu_cog = 0.48 #[m] displacement of cog to vmu wrt vmu

    if name in ['pose x', 'pose y']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][1]
        y = self.availableData[availableDataList.index(nameDependency[0])][2]
        theta = self.availableData[availableDataList.index(nameDependency[1])][2]

        if name == 'pose x':
            y = y + vmu_cog * np.cos(theta)
        else:
            y = y + vmu_cog * np.sin(theta)

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])

    differentiate = 'pose vx', 'pose vy', 'pose vtheta', 'pose atheta'
    if name in differentiate:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0][0]
        index = availableDataList.index(nameDependency)
        x = self.availableData[index][1]
        y = self.availableData[index][2]
        t, dydt = derivative_X_dX(name, x, y)
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = dydt
        else:
            self.availableData.append([name, t, dydt])

    if name in ['pose ax', 'pose ay']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[1])][1]
        y = self.availableData[availableDataList.index(nameDependency[1])][2]
        t, dydt = derivative_X_dX(name, x, y)
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = dydt
        else:
            self.availableData.append([name, t, dydt])

    if name == 'xy trace':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][2]
        y = self.availableData[availableDataList.index(nameDependency[1])][2]

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][1] = x
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])

    if name == 'xy trace atvmu':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][2]
        y = self.availableData[availableDataList.index(nameDependency[1])][2]

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][1] = x
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])
    
    if name == 'vehicle slip angle':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[3])][1]
        vx = self.availableData[availableDataList.index(nameDependency[3])][2]
        theta = self.availableData[availableDataList.index(nameDependency[0])][2]
        vy = self.availableData[availableDataList.index(nameDependency[4])][2]

        y = theta[:-1] - np.arctan2(vy, vx)

        for i in range(len(y)):
            if y[i] < -np.pi:
                y[i] = y[i] + 2 * np.pi
            if y[i] > np.pi:
                y[i] = y[i] - 2 * np.pi

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])

    if name in ['vmu ax', 'vmu ay']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][1]
        a = self.availableData[availableDataList.index(nameDependency[0])][2]
        atheta_t = self.availableData[availableDataList.index(nameDependency[3])][1]
        atheta = self.availableData[availableDataList.index(nameDependency[3])][2]

        while x[0] < atheta_t.iloc[0]:
            # x.pop(0)
            x = np.delete(x, 0)
            a = np.delete(a, 0)
        while x[-1] > atheta_t.iloc[-1]:
            # x.pop()
            x = np.delete(x, -1)
            a = np.delete(a, -1)

        interp = interp1d(atheta_t, atheta)
        atheta = interp(x)

        if name == 'vmu ax':
            y = a
        else:
            y = a - atheta * vmu_cog

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])

    if name in ['vehicle vx', 'vehicle vy']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[2])][1]
        vx = self.availableData[availableDataList.index(nameDependency[2])][2]
        vy = self.availableData[availableDataList.index(nameDependency[3])][2]
        slipAngle = self.availableData[availableDataList.index(nameDependency[4])][2]

        if name == 'vehicle vx':
            y = np.sqrt(vx ** 2 + vy ** 2) * np.cos(slipAngle)
        else:
            y = np.sqrt(vx ** 2 + vy ** 2) * np.sin(slipAngle)

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])

    if name in ['vehicle ax total', 'vehicle ay total']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[7])][1]
        vx = self.availableData[availableDataList.index(nameDependency[7])][2]
        vy = self.availableData[availableDataList.index(nameDependency[8])][2]
        vtheta = self.availableData[availableDataList.index(nameDependency[3])][2]
        if name == 'vehicle ax total':
            t, dydt = derivative_X_dX(name, x, vx)
            y = dydt - (vtheta * vy)[:-1]
        else:
            t, dydt = derivative_X_dX(name, x, vy)
            y = dydt + (vtheta * vx)[:-1]

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x[:-1], y])

    if name in ['vehicle ax only transl', 'vehicle ay only transl']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[5])][1]
        ax = self.availableData[availableDataList.index(nameDependency[5])][2]
        ay = self.availableData[availableDataList.index(nameDependency[6])][2]
        theta = self.availableData[availableDataList.index(nameDependency[0])][2]
        if name == 'vehicle ax only transl':
            y = ax * np.cos(theta[:-2]) + ay * np.sin(theta[:-2])
        else:
            y = ay * np.cos(theta[:-2]) - ax * np.sin(theta[:-2])

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])
            
    if name in ['MH power accel rimo left', 'MH power accel rimo right']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = list(self.availableData[availableDataList.index(nameDependency[0])][1].values)
        motorPower = list(self.availableData[availableDataList.index(nameDependency[0])][2])
        velocity_t = self.availableData[availableDataList.index(nameDependency[6])][1]
        velocity = self.availableData[availableDataList.index(nameDependency[6])][2]

        while x[0] < velocity_t.iloc[0]:
            x.pop(0)
            motorPower.pop(0)
        while x[-1] > velocity_t.iloc[-1]:
            x.pop()
            motorPower.pop()
        interp = interp1d(velocity_t, velocity)
        velocity = interp(x)

        lookupFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/dataanalysis/pyqtgraph/lookup_cur_vel_to_acc.pkl'   #lookupTable file
        try:
            with open(lookupFilePath, 'rb') as f:
                lookupTable = pickle.load(f)
            print('lookup_cur_vel_to_acc file for preprocessing located and opened.')
        except:
            print('lookup_cur_vel_to_acc file for preprocessing does not exist. Creating file...')
            lookupTable = pd.DataFrame()
        interp = interp2d(lookupTable.columns, lookupTable.index, lookupTable.values)
        powerAcceleration = [float(interp(XX,YY)) for XX,YY in zip(velocity,motorPower)]

        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = powerAcceleration
        else:
            self.availableData.append([name, x, powerAcceleration])
    
    if name == 'MH AB':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[7])][1]
        powerAccelL = self.availableData[availableDataList.index(nameDependency[7])][2]
        powerAccelR = self.availableData[availableDataList.index(nameDependency[8])][2]
        brakePos_t = self.availableData[availableDataList.index(nameDependency[0])][1]
        brakePos = self.availableData[availableDataList.index(nameDependency[0])][2]
        powerAccel = np.dstack((powerAccelL,powerAccelR))
        
        AB_rimo = np.mean(powerAccel, axis=2)
        
        while x[0] < brakePos_t.iloc[0]:
            x.pop(0)
        while x[-1] > brakePos_t.iloc[-1]:
            x.pop()
            
        interp1 = interp1d(brakePos_t, brakePos)
        brakePos = interp1(x)
        
        staticBrakeFunctionFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/dataanalysis/pyqtgraph/staticBrakeFunction.pkl'   #static brake function file
        try:
            with open(staticBrakeFunctionFilePath, 'rb') as f:
                staticBrakeFunction = pickle.load(f)
            print('staticBrakeFunction file for preprocessing located and opened.')
        except:
            print('staticBrakeFunction file for preprocessing does not exist. Creating file...')
            staticBrakeFunction = pd.DataFrame()
        
        interp2 = interp1d(staticBrakeFunction['brake pos'], staticBrakeFunction['deceleration'])
        deceleration = interp2(brakePos)
        
        AB_brake = -deceleration
        
        AB_tot = AB_rimo + AB_brake
        
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = AB_tot[0,:]
        else:
            self.availableData.append([name, x, AB_tot[0,:]])
            
    if name == 'MH TV':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[6])][1]
        powerAccelL = self.availableData[availableDataList.index(nameDependency[6])][2]
        powerAccelR = self.availableData[availableDataList.index(nameDependency[7])][2]
        
        TV = np.subtract(powerAccelR, powerAccelL)/2.0
        print(TV)
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = TV
        else:
            self.availableData.append([name, x, TV])
            
    if name == 'MH BETA':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][1]
        steerCal = self.availableData[availableDataList.index(nameDependency[0])][2]
        
        print(-0.63*steerCal[0]*steerCal[0]*steerCal[0]+0.94*steerCal[0])
        
        beta = -0.63*steerCal*steerCal*steerCal+0.94*steerCal
        
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = beta
        else:
            self.availableData.append([name, x, beta])

def derivative_X_dX(name, x, y):
    if name == 'pose vtheta':
        dydx = np.diff(y) / np.diff(x)
        lim = 4
        for i in range(len(dydx) - 1):
            if dydx[i] > lim:
                dydx[i] = (y[i + 1] - y[i] - 2 * np.pi) / (x[i + 1] - x[i])
            elif dydx[i] < -lim:
                dydx[i] = (y[i + 1] - y[i] + 2 * np.pi) / (x[i + 1] - x[i])
                # k = 1
                # while dydx[i + k] > lim or dydx[i + k] < -lim:
                #     k += 1
                # for l in range(k):
                #     dydx[i + l] = (dydx[i + l - 1] + dydx[i + k]) / 2.0

    else:
        dydx = np.diff(y) / np.diff(x)
    return x[:-1], dydx
