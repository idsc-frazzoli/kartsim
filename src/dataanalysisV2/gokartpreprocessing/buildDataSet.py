#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
from dataanalysisV2.dataIO import getDirectories
from dataanalysisV2.gokartpreprocessing.preprocessing import updateData
from dataanalysisV2.gokartpreprocessing.importdata import setListItems

    
def stirData(pathRootData, preproParams, requiredList, nono_list):
    loggoodcount = 0
    logtotcount = 0
    for day in preproParams:
        for log in preproParams[day]:
            loggoodcount += 1
            logtotcount += 1
            for topic in preproParams[day][log]:
                if (topic in requiredList and int(preproParams[day][log][topic]) == 0) or (topic in nono_list and int(preproParams[day][log][topic]) == 1):
                    preproParams[day][log]['goodData'] = 0
                    loggoodcount -= 1
                    break
    print(loggoodcount, 'of', logtotcount, 'logs are used for creating this dataset.')

    comp_tot = 0
    testDays = getDirectories(pathRootData)
    testDays.sort()
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()
        for logNr in logNrs:
            if preproParams[testDay][logNr]['goodData']:
                comp_tot += 1

    kartDataAll = {}
    skipCount = 0
    comp_count = 0
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
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
    return kartDataAll
