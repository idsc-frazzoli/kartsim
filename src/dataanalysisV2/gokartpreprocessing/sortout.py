#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
import numpy as np
import time

from dataanalysisV2.dataIO import getDirectories
from dataanalysisV2.gokartpreprocessing.preprocessing import updateData
from dataanalysisV2.gokartpreprocessing.importdata import setListItems


def sort_out(pathRootData, preproParams, redo):
    t = time.time()

    path_logs, comp_tot = initialize_parameters(pathRootData, preproParams, redo)

    comp_count = 0
    for testDay, logNr, path in path_logs:

        kartData, allDataNames = setListItems(path)
        vmuDataNames = ['pose x [m]', 'pose y [m]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]', 'pose atheta [rad*s^-2]',
                        'vmu ax [m*s^-2] [m*s^-2]', 'vmu ay [m*s^-2] [m*s^-2]']
        kartData = updateData(kartData, vmuDataNames)

        statusInfo = logNr + ':  '

        #___distance
        dx = np.subtract(kartData['pose x [m]']['data'][1][1:], kartData['pose x [m]']['data'][1][:-1])
        dy = np.subtract(kartData['pose y [m]']['data'][1][1:], kartData['pose y [m]']['data'][1][:-1])
        dist = np.sum(np.sqrt(np.square(dx) + np.square(dy)))
        if dist > 100:
            preproParams[testDay][logNr]['multiple laps'] = 1
        else:
            preproParams[testDay][logNr]['multiple laps'] = 0

        #___driving style
        if np.std(kartData['vehicle vy [m*s^-1]']['data'][1]) > 0.2:
            preproParams[testDay][logNr]['high slip angles'] = 1
        else:
            preproParams[testDay][logNr]['high slip angles'] = 0

        #___reverse
        if np.min(kartData['vehicle vx [m*s^-1]']['data'][1]) < -0.2:
            preproParams[testDay][logNr]['reverse'] = 1
        else:
            preproParams[testDay][logNr]['reverse'] = 0

        #___steering
        lenSteerCmd = len(kartData['steer torque cmd [n.a.]']['data'][1])
        if lenSteerCmd == 1:
            statusInfo = statusInfo + 'steering cmd data missing,  '
            preproParams[testDay][logNr]['steer torque cmd [n.a.]'] = 0
        elif kartData['steer torque cmd [n.a.]']['data'][1][int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)].count(0) / len(
                kartData['steer torque cmd [n.a.]']['data'][1][int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)]) > 0.05:
            statusInfo = statusInfo + 'steering cmd data: too many zeros...,  '
            preproParams[testDay][logNr]['steer torque cmd [n.a.]'] = 0
        elif np.abs(np.mean(kartData['steer torque cmd [n.a.]']['data'][1])) < 0.01 and np.std(
                kartData['steer torque cmd [n.a.]']['data'][1]) < 0.1:
            statusInfo = statusInfo + 'steering cmd data insufficient,  '
            preproParams[testDay][logNr]['steer torque cmd [n.a.]'] = 0
        else:
            preproParams[testDay][logNr]['steer torque cmd [n.a.]'] = 1

        lenSteerPos = len(kartData['steer position cal [n.a.]']['data'][1])
        if lenSteerPos == 1:
            statusInfo = statusInfo + 'steering pos cal data missing,  '
            preproParams[testDay][logNr]['steer position cal [n.a.]'] = 0
        elif kartData['steer position cal [n.a.]']['data'][1][int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)].count(
                0) / len(
                kartData['steer position cal [n.a.]']['data'][1][int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)]) > 0.05:
            statusInfo = statusInfo + 'steering pos cal data: too many zeros...,  '
            preproParams[testDay][logNr]['steer position cal [n.a.]'] = 0
        elif np.abs(np.mean(kartData['steer position cal [n.a.]']['data'][1])) < 0.01 and np.std(
                kartData['steer position cal [n.a.]']['data'][1]) < 0.1:
            statusInfo = statusInfo + 'steering pos cal data missing or insufficient,  '
            preproParams[testDay][logNr]['steer position cal [n.a.]'] = 0
        else:
            preproParams[testDay][logNr]['steer position cal [n.a.]'] = 1

        #___brake
        if np.max(kartData['brake position cmd [m]']['data'][1]) < 0.025 and np.mean(kartData['brake position cmd [m]']['data'][1]) < 0.004:
            statusInfo = statusInfo + 'brake position cmd [m] data missing or insufficient,  '
            preproParams[testDay][logNr]['brake position cmd [m]'] = 0
        else:
            preproParams[testDay][logNr]['brake position cmd [m]'] = 1

        if np.max(kartData['brake position effective [m]']['data'][1]) < 0.025 and np.mean(kartData['brake position effective [m]']['data'][1]) < 0.004:
            statusInfo = statusInfo + 'brake position effective [m] data missing or insufficient,  '
            preproParams[testDay][logNr]['brake position effective [m]'] = 0
        else:
            preproParams[testDay][logNr]['brake position effective [m]'] = 1

        #___VMU
        if np.abs(np.mean(kartData['vmu ax [m*s^-2]']['data'][1])) < 0.01 and np.std(kartData['vmu ax [m*s^-2]']['data'][1]) < 0.01:
            statusInfo = statusInfo + 'vmu ax [m*s^-2] data missing or insufficient,  '
            preproParams[testDay][logNr]['vmu ax [m*s^-2]'] = 0
        else:
            preproParams[testDay][logNr]['vmu ax [m*s^-2]'] = 1

        if np.abs(np.mean(kartData['vmu ay [m*s^-2]']['data'][1])) < 0.01 and np.std(kartData['vmu ay [m*s^-2]']['data'][1]) < 0.05:
            statusInfo = statusInfo + 'vmu ay [m*s^-2] data missing or insufficient,  '
            preproParams[testDay][logNr]['vmu ay [m*s^-2]'] = 0
        else:
            preproParams[testDay][logNr]['vmu ay [m*s^-2]'] = 1

        if np.abs(np.mean(kartData['vmu vtheta [rad*s^-1]']['data'][1])) < 0.01 and np.std(
                kartData['vmu vtheta [rad*s^-1]']['data'][1]) < 0.05:
            statusInfo = statusInfo + 'vmu vtheta [rad*s^-1] data missing or insufficient,  '
            preproParams[testDay][logNr]['vmu vtheta [rad*s^-1]'] = 0
        else:
            preproParams[testDay][logNr]['vmu vtheta [rad*s^-1]'] = 1

        #___MH model specific
        if preproParams[testDay][logNr]['brake position effective [m]']:
            preproParams[testDay][logNr]['MH AB [m*s^-2]'] = 1
            preproParams[testDay][logNr]['MH TV [rad*s^-2]'] = 1
        else:
            preproParams[testDay][logNr]['MH AB [m*s^-2]'] = 0
            preproParams[testDay][logNr]['MH TV [rad*s^-2]'] = 0

        if preproParams[testDay][logNr]['steer position cal [n.a.]']:
            preproParams[testDay][logNr]['MH BETA [rad]'] = 1
        else:
            preproParams[testDay][logNr]['MH BETA [rad]'] = 0

        statusInfo = statusInfo + 'done'
        comp_count += 1
        print(statusInfo)
        print(str(int(comp_count / comp_tot * 100)), '% completed.  elapsed time:', int(time.time() - t), "s", end='\r')

    return preproParams


def initialize_parameters(pathRootData, preproParams, redo):
    testDays = getDirectories(pathRootData)
    testDays.sort()

    path_logs = ()
    comp_tot = 0
    for testDay in testDays:
        if testDay not in preproParams:
            preproParams[testDay] = {}
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()
        comp_tot += len(logNrs)

        for logNr in logNrs:
            if logNr in preproParams[testDay] and not redo:
                print(logNr, ' already done. Continuing with next logNr')
                continue
            if logNr not in preproParams[testDay]:
                preproParams[testDay][logNr] = {}
                preproParams[testDay][logNr]['goodData'] = 1
            path_logs = path_logs + ((testDay, logNr, pathTestDay + '/' + logNr),)

    return path_logs, comp_tot