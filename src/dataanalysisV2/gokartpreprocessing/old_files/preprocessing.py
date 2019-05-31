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
    vmu_cog = 0.46  # [m] displacement of cog to vmu wrt vmu

    differentiate = 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'pose ax [m*s^-2]', 'pose ay [m*s^-2]', 'pose atheta [rad*s^-2]'
    differentiateFrom = 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'pose vtheta [rad*s^-1]'

    if name in differentiate:
        index = differentiate.index(name)
        nameFrom = differentiateFrom[index]
        t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                  kartData[nameFrom]['data'][1])
        kartData[name]['data'] = [list(t), list(dydt)]

    if name in ['pose x [m]', 'pose y [m]']:

        theta = kartData['pose theta [rad]']['data'][1]

        if name == 'pose x [m]':
            x = kartData['pose x atvmu [m]']['data'][0]
            y = kartData['pose x atvmu [m]']['data'][1]
            y = y + vmu_cog * np.cos(theta)
        else:
            x = kartData['pose y atvmu [m]']['data'][0]
            y = kartData['pose y atvmu [m]']['data'][1]
            y = y + vmu_cog * np.sin(theta)

        kartData[name]['data'] = [x, y]


    elif name == 'vehicle slip angle [rad]':
        x = kartData['pose vx [m*s^-1]']['data'][0]
        vx = kartData['pose vx [m*s^-1]']['data'][1]
        theta = kartData['pose theta [rad]']['data'][1]
        vy = kartData['pose vy [m*s^-1]']['data'][1]
        y = theta[:-1] - np.arctan2(vy, vx)

        for i in range(len(y)):
            if y[i] < -np.pi:
                y[i:] = np.add(y[i:], 2 * np.pi)
            if y[i] > np.pi:
                y[i:] = np.subtract(y[i:], 2 * np.pi)

        kartData[name]['data'] = [x, y]

    elif name in ['vmu ax [m*s^-2]', 'vmu ay [m*s^-2]']:
        if name == 'vmu ax [m*s^-2]':
            x = kartData['vmu ax atvmu (forward) [m*s^-2]']['data'][0]
            a = kartData['vmu ax atvmu (forward) [m*s^-2]']['data'][1]
        else:
            x = kartData['vmu ay atvmu (left)[m*s^-2]']['data'][0]
            a = kartData['vmu ay atvmu (left)[m*s^-2]']['data'][1]
        atheta_t = kartData['pose atheta [rad*s^-2]']['data'][0]
        atheta = kartData['pose atheta [rad*s^-2]']['data'][1]

        while x[0] < atheta_t[0]:
            x = np.delete(x, 0)
            a = np.delete(a, 0)
        while x[-1] > atheta_t[-1]:
            x = np.delete(x, -1)
            a = np.delete(a, -1)

        interp = interp1d(atheta_t, atheta)
        atheta = interp(x)

        if name == 'vmu ax [m*s^-2]':
            y = a
        else:
            y = a - atheta * vmu_cog

        kartData[name]['data'] = [x, y]

    elif name in ['vehicle ax total [m*s^-2]', 'vehicle ay total [m*s^-2]']:
        x = kartData['vehicle vx [m*s^-1]']['data'][0]
        vx = kartData['vehicle vx [m*s^-1]']['data'][1]
        vy = kartData['vehicle vy [m*s^-1]']['data'][1]
        vtheta = kartData['pose vtheta [rad*s^-1]']['data'][1]

        if name == 'vehicle ax total [m*s^-2]':
            nameFrom = 'vehicle vx [m*s^-1]'
            t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                      kartData[nameFrom]['data'][1])
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vy):
                vy = vy[:-1]

            y = dydt + (vtheta * vy)
        else:
            nameFrom = 'vehicle vy [m*s^-1]'
            t, dydt = derivative_X_dX(name, kartData[nameFrom]['data'][0],
                                      kartData[nameFrom]['data'][1])
            while len(dydt) < len(vtheta):
                vtheta = vtheta[:-1]
            while len(dydt) < len(vx):
                vx = vx[:-1]

            y = dydt - (vtheta * vx)

        kartData[name]['data'] = [x[:-1], y]

    elif name in ['vehicle ax only transl [m*s^-2]', 'vehicle ay only transl [m*s^-2]']:
        x = kartData['pose ax [m*s^-2]']['data'][0]
        ax = kartData['pose ax [m*s^-2]']['data'][1]
        ay = kartData['pose ay [m*s^-2]']['data'][1]
        theta = kartData['pose theta [rad]']['data'][1]
        if name == 'vehicle ax only transl [m*s^-2]':
            y = ax * np.cos(theta[:-2]) + ay * np.sin(theta[:-2])
        else:
            y = ay * np.cos(theta[:-2]) - ax * np.sin(theta[:-2])

        kartData[name]['data'] = [x, y]

    elif name in ['vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]']:
        if name == 'vehicle ax local [m*s^-2]':
            t = kartData['vehicle vx [m*s^-1]']['data'][0]
            v = kartData['vehicle vx [m*s^-1]']['data'][1]
        else:
            t = kartData['vehicle vy [m*s^-1]']['data'][0]
            v = kartData['vehicle vy [m*s^-1]']['data'][1]

        t, dydt = derivative_X_dX(name, t, v)

        kartData[name]['data'] = [t, dydt]

    elif name in ['MH power accel rimo left [m*s^-2]', 'MH power accel rimo right [m*s^-2]']:
        if name == 'MH power accel rimo left [m*s^-2]':
            x = kartData['motor torque cmd left [A_rms]']['data'][0]
            motorPower = kartData['motor torque cmd left [A_rms]']['data'][1]
        else:
            x = kartData['motor torque cmd right [A_rms]']['data'][0]
            motorPower = kartData['motor torque cmd right [A_rms]']['data'][1]
        velocity_t = kartData['vehicle vx [m*s^-1]']['data'][0]
        velocity = kartData['vehicle vx [m*s^-1]']['data'][1]

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

    elif name == 'MH AB [m*s^-2]':
        x = kartData['MH power accel rimo left [m*s^-2]']['data'][0]
        powerAccelL = kartData['MH power accel rimo left [m*s^-2]']['data'][1]
        powerAccelR = kartData['MH power accel rimo right [m*s^-2]']['data'][1]
        brakePos_t = kartData['brake position effective [m]']['data'][0]
        brakePos = kartData['brake position effective [m]']['data'][1]
        velx_t = kartData['vehicle vx [m*s^-1]']['data'][0]
        velx = kartData['vehicle vx [m*s^-1]']['data'][1]

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

    elif name == 'MH TV [rad*s^-2]':
        x = kartData['MH power accel rimo left [m*s^-2]']['data'][0]
        powerAccelL = kartData['MH power accel rimo left [m*s^-2]']['data'][1]
        powerAccelR = kartData['MH power accel rimo right [m*s^-2]']['data'][1]

        TV = np.subtract(powerAccelR, powerAccelL) / 2.0

        kartData[name]['data'] = [x, TV]

    elif name == 'MH BETA [rad]':
        x = kartData['steer position cal [n.a.]']['data'][0]
        steerCal = np.array(kartData['steer position cal [n.a.]']['data'][1])
        BETA = -0.625 * np.power(steerCal, 3) + 0.939 * steerCal

        kartData[name]['data'] = [x, BETA]
    return kartData