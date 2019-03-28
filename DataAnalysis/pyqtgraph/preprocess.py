#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:37:23 2019

@author: mvb
"""
from pyqtgraph.Qt import QtCore
import numpy as np
from scipy.interpolate import interp1d


def interpolation(self, x, y, timeStep):
    interp = interp1d(x,y)
    
    if x[0]%timeStep != 0:
        xBegin = (timeStep - x[0]%timeStep) + x[0]
    else:
        xBegin = x[0]
        
    if x.iloc[-1]%timeStep != 0:
        xStop = x.iloc[-1] - x.iloc[-1]%timeStep
    else:
        xStop = x.iloc[-1]
        
    xInterp = np.arange(xBegin, xStop, timeStep)
    yInterp = interp(xInterp)
    return xInterp, yInterp

def preProcessing(self, name):
    availableDataList = [item[0] for item in self.availableData]
    differentiate = 'pose vx', 'pose vy', 'pose vtheta', 'pose ax', 'pose ay', 'pose atheta' 
    if name in differentiate:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0][0]
        t, dydt = derivative_X_dX(self, nameDependency, name)
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = dydt
        else:
            self.availableData.append([name, t, dydt])
        
    if name == 'vehicle slip angle':
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[1])][1]
        vx = self.availableData[availableDataList.index(nameDependency[1])][2]
        theta = self.availableData[availableDataList.index(nameDependency[0])][2]
        vy = self.availableData[availableDataList.index(nameDependency[2])][2]
        
        y = theta[:-1] - np.arctan2(vy,vx)
        
        for i in range(len(y)):
            if y[i] < -np.pi:
                y[i] = y[i] + 2*np.pi
            if y[i] > np.pi:
                y[i] = y[i] - 2*np.pi
        
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])
    
    if name in ['vehicle vx', 'vehicle vy']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[0])][1]
        vx = self.availableData[availableDataList.index(nameDependency[0])][2]
        vy = self.availableData[availableDataList.index(nameDependency[1])][2]
        slipAngle = self.availableData[availableDataList.index(nameDependency[2])][2]
        
        if name == 'vehicle vx':
            y = np.sqrt(vx**2 + vy**2) * np.cos(slipAngle)
        else:
            y = np.sqrt(vx**2 + vy**2) * np.sin(slipAngle)
        
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x, y])
            
    if name in ['vehicle ax total', 'vehicle ay total']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[5])][1]
        vx = self.availableData[availableDataList.index(nameDependency[5])][2]
        vy = self.availableData[availableDataList.index(nameDependency[6])][2]
        vtheta = self.availableData[availableDataList.index(nameDependency[1])][2]
        if name == 'vehicle ax total':
            t, dydt = derivative_X_dX(self, nameDependency[5], name)
            y = dydt - (vtheta * vy)[:-1]
        else:
            t, dydt = derivative_X_dX(self, nameDependency[6], name)
            y = dydt + (vtheta * vx)[:-1]
        
        if name in availableDataList:
            index = availableDataList.index(name)
            self.availableData[index][2] = y
        else:
            self.availableData.append([name, x[:-1], y])
    
    if name in ['vehicle ax only transl', 'vehicle ay only transl']:
        for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
            nameDependency = item.dependencies[0]
        x = self.availableData[availableDataList.index(nameDependency[3])][1]
        ax = self.availableData[availableDataList.index(nameDependency[3])][2]
        ay = self.availableData[availableDataList.index(nameDependency[4])][2]
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
    
def derivative_X_dX(self, nameSource, nameProduct):
    availableDataList = [item[0] for item in self.availableData]
    index = availableDataList.index(nameSource)
    x = self.availableData[index][1]
    y = self.availableData[index][2]
    if nameSource == 'pose theta':
        dydx = np.diff(y)/np.diff(x)
        lim = 4
#        while max(dydx) > 2.0 or min(dydx) < 2.0:
#        for j in range(100):
#            print(max(dydx), min(dydx))
#            for i in range(len(dydx)-1):
#                if dydx[i]>2.0 or dydx[i]<-2.0:
#                    dydx[i] = (dydx[i-1] + dydx[i+1])/2.0
        for i in range(len(dydx)-1):
            if dydx[i]>lim or dydx[i]<-lim:
                k=1
                while dydx[i+k]>lim or dydx[i+k]<-lim:
                    k+=1
                for l in range(k):
                    dydx[i+l] = (dydx[i+l-1] + dydx[i+k])/2.0

    else:
        dydx = np.diff(y)/np.diff(x)
    return x[:-1], dydx


