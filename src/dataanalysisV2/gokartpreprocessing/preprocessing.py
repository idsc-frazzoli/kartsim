#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:29

@author: mvb
"""
from dataanalysisV2.mathfunction import derivative_X_dX

import numpy as np
import pandas as pd
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d, interp2d


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
    vmu_cog = 0.48  # [m] displacement of cog to vmu wrt vmu

    differentiate = 'pose vx', 'pose vy', 'pose vtheta', 'pose ax', 'pose ay', 'pose atheta'
    differentiateFrom = 'pose x', 'pose y', 'pose theta', 'pose vx', 'pose vy', 'pose vtheta'

    if name in differentiate:
        index = differentiate.index(name)
        nameFrom = differentiateFrom[index]
        t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
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
                y[i:] = np.add(y[i:], 2 * np.pi)
            if y[i] > np.pi:
                y[i:] = np.subtract(y[i:], 2 * np.pi)

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

    elif name in ['vehicle ax total', 'vehicle ay total']:
        x = kartData['vehicle vx']['data'][0]
        vx = kartData['vehicle vx']['data'][1]
        vy = kartData['vehicle vy']['data'][1]
        vtheta = kartData['pose vtheta']['data'][1]

        if name == 'vehicle ax total':
            nameFrom = 'vehicle vx'
            t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                      kartData[nameFrom]['data'][1])
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vy):
                vy = vy[:-1]

            y = dydt + (vtheta * vy)
        else:
            nameFrom = 'vehicle vy'
            t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                      kartData[nameFrom]['data'][1])
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vx):
                vx = vx[:-1]

            y = dydt - (vtheta * vx)

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

    elif name in ['vehicle ax local', 'vehicle ay local']:
        if name == 'vehicle ax local':
            t = kartData['vehicle vx']['data'][0]
            v = kartData['vehicle vx']['data'][1]
        else:
            t = kartData['vehicle vy']['data'][0]
            v = kartData['vehicle vy']['data'][1]

        t, dydt = derivative_X_dX(name, t, v)

        kartData[name]['data'] = [t, dydt]

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
            x = np.delete(x, 0)
            motorPower = np.delete(motorPower, 0)
        while x[-1] > velocity_t[-1]:
            x = np.delete(x, -1)
            motorPower = np.delete(motorPower, -1)
        interp = interp1d(velocity_t, velocity)
        velocity = interp(x)

        lookupFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/dataanalysisV2/lookupfunctions/lookup_cur_vel_to_acc.pkl'  # lookupTable file
        try:
            with open(lookupFilePath, 'rb') as f:
                lookupTable = pickle.load(f)
            # print('Lookup Table file for preprocessing located and opened.')
        except:
            print('Lookup Table file for preprocessing does not exist. Creating file...')
            lookupTable = pd.DataFrame()
        interp = interp2d(lookupTable.columns, lookupTable.index, lookupTable.values)
        powerAcceleration = [float(interp(XX, YY)) for XX, YY in zip(velocity, motorPower)]

        kartData[name]['data'] = [x, powerAcceleration]

    elif name == 'MH AB':
        x = kartData['MH power accel rimo left']['data'][0]
        powerAccelL = kartData['MH power accel rimo left']['data'][1]
        powerAccelR = kartData['MH power accel rimo right']['data'][1]
        brakePos_t = kartData['brake position effective']['data'][0]
        brakePos = kartData['brake position effective']['data'][1]
        velx_t = kartData['vehicle vx']['data'][0]
        velx = kartData['vehicle vx']['data'][1]

        powerAccel = np.dstack((powerAccelL, powerAccelR))

        AB_rimo = np.mean(powerAccel, axis=2)

        while x[0] < brakePos_t[0]:
            x = np.delete(x, 0)
            AB_rimo = AB_rimo[:, 1:]
        while x[-1] > brakePos_t[-1]:
            x = np.delete(x, -1)
            AB_rimo = AB_rimo[:, :-1]
        while x[0] < velx_t[0]:
            x = np.delete(x, 0)
            AB_rimo = AB_rimo[:, 1:]
        while x[-1] > velx_t[-1]:
            x = np.delete(x, -1)
            AB_rimo = AB_rimo[:, :-1]

        interp1B = interp1d(brakePos_t, brakePos)
        brakePos = interp1B(x)
        interp1V = interp1d(velx_t, velx)
        velx = interp1V(x)

        staticBrakeFunctionFilePath = '/home/mvb/0_ETH/01_MasterThesis/kartsim/src/dataanalysisV2/lookupfunctions/staticBrakeFunction.pkl'  # static brake function file
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

        kartData[name]['data'] = [x, AB_tot[0, :]]

    elif name == 'MH TV':
        x = kartData['MH power accel rimo left']['data'][0]
        powerAccelL = kartData['MH power accel rimo left']['data'][1]
        powerAccelR = kartData['MH power accel rimo right']['data'][1]

        TV = np.subtract(powerAccelR, powerAccelL) / 2.0

        kartData[name]['data'] = [x, TV]

    elif name == 'MH BETA':
        x = kartData['steer position cal']['data'][0]
        steerCal = np.array(kartData['steer position cal']['data'][1])
        BETA = -0.625 * np.power(steerCal, 3) + 0.939 * steerCal

        kartData[name]['data'] = [x, BETA]
    return kartData