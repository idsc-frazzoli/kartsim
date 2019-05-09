#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
import numpy as np

from dataanalysisV2.dataIO import getDirectories
from dataanalysisV2.gokartpreprocessing.importdata import setListItems

def sort_out(pathRootData, preproParams, redo):
    testDays = getDirectories(pathRootData)
    testDays.sort()

    comp_tot = 0
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        comp_tot += len(logNrs)

    comp_count = 0
    for testDay in testDays:
        if testDay not in preproParams:
            preproParams[testDay] = {}
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
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
            if kartData['steer torque cmd']['data'][1][int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)].count(0) / len(
                    kartData['steer torque cmd']['data'][1][int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)]) > 0.05:
                statusInfo = statusInfo + 'steering cmd data: too many zeros...,  '
                preproParams[testDay][logNr]['steer torque cmd'] = 0
            elif np.abs(np.mean(kartData['steer torque cmd']['data'][1])) < 0.01 and np.std(
                    kartData['steer torque cmd']['data'][1]) < 0.1:
                statusInfo = statusInfo + 'steering cmd data missing or insufficient,  '
                preproParams[testDay][logNr]['steer torque cmd'] = 0
            else:
                preproParams[testDay][logNr]['steer torque cmd'] = 1

            lenSteerPos = len(kartData['steer position cal']['data'][1])
            if kartData['steer position cal']['data'][1][int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)].count(
                    0) / len(
                    kartData['steer position cal']['data'][1][int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)]) > 0.05:
                statusInfo = statusInfo + 'steering pos cal data: too many zeros...,  '
                preproParams[testDay][logNr]['steer position cal'] = 0
            elif np.abs(np.mean(kartData['steer position cal']['data'][1])) < 0.01 and np.std(
                    kartData['steer position cal']['data'][1]) < 0.1:
                statusInfo = statusInfo + 'steering pos cal data missing or insufficient,  '
                preproParams[testDay][logNr]['steer position cal'] = 0
            else:
                preproParams[testDay][logNr]['steer position cal'] = 1

            if np.max(kartData['brake position cmd']['data'][1]) < 0.025 or np.std(
                    kartData['brake position cmd']['data'][1]) < 0.001:
                statusInfo = statusInfo + 'brake position cmd data missing or insufficient,  '
                print(np.abs(np.mean(kartData['brake position cmd']['data'][1])))
                print(np.std(kartData['brake position cmd']['data'][1]))
                preproParams[testDay][logNr]['brake position cmd'] = 0
            else:
                print(np.abs(np.mean(kartData['brake position cmd']['data'][1])))
                print(np.std(kartData['brake position cmd']['data'][1]))
                preproParams[testDay][logNr]['brake position cmd'] = 1

            if np.max(kartData['brake position effective']['data'][1]) < 0.025 or np.std(
                    kartData['brake position effective']['data'][1]) < 0.001:
                statusInfo = statusInfo + 'brake position effective data missing or insufficient,  '
                print(np.abs(np.mean(kartData['brake position effective']['data'][1])))
                print(np.std(kartData['brake position effective']['data'][1]))
                preproParams[testDay][logNr]['brake position effective'] = 0
            else:
                print(np.abs(np.mean(kartData['brake position effective']['data'][1])))
                print(np.std(kartData['brake position effective']['data'][1]))
                preproParams[testDay][logNr]['brake position effective'] = 1

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

            if np.abs(np.mean(kartData['vmu vtheta']['data'][1])) < 0.01 and np.std(
                    kartData['vmu vtheta']['data'][1]) < 0.05:
                statusInfo = statusInfo + 'vmu vtheta data missing or insufficient,  '
                preproParams[testDay][logNr]['vmu vtheta'] = 0
            else:
                preproParams[testDay][logNr]['vmu vtheta'] = 1

            if preproParams[testDay][logNr]['brake position effective']:
                preproParams[testDay][logNr]['MH AB'] = 1
                preproParams[testDay][logNr]['MH TV'] = 1
            else:
                preproParams[testDay][logNr]['MH AB'] = 0
                preproParams[testDay][logNr]['MH TV'] = 0

            if preproParams[testDay][logNr]['steer position cal']:
                preproParams[testDay][logNr]['MH BETA'] = 1
            else:
                preproParams[testDay][logNr]['MH BETA'] = 0

            statusInfo = statusInfo + 'done'
            comp_count += 1
            print(statusInfo)
            print(str(int(comp_count / comp_tot * 100)), '% completed.  ', statusInfo, end='\r')

    return preproParams