#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
from dataanalysisV2.data_io import getDirectories
from dataanalysisV2.gokartpreprocessing.preprocessing import filter_raw_data

from dataanalysisV2.gokartpreprocessing.gokart_raw_data import GokartRawData
    
def prepare_dataset(pathRootData, data_tags, required_data_list, required_tags_list):

    nr_of_good_logs = 0
    total_nr_of_logs = 0
    testDays = getDirectories(pathRootData)
    testDays.sort()
    for testDay in testDays[-2:-1]:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()
        total_nr_of_logs += len(logNrs)
        for logNr in logNrs:
            if data_tags.good_data(testDay,logNr):
                nr_of_good_logs += 1
    test_log_pairs = get_testday_lognr_pairs(pathRootData)
    total_nr_of_logs = len(test_log_pairs)
    good_logs_list = [data_tags.good_data(day, log) for day, log in test_log_pairs]
    nr_of_good_logs = sum(good_logs_list)

    print(nr_of_good_logs, 'of', total_nr_of_logs, 'logs contain the necessary tags.')

    filtered_kart_data = {}
    skipCount = 0
    comp_count = 0

    raw_data = GokartRawData(required_data_list=required_data_list)

    for testDay in testDays[-2:-1]:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()

        for logNr in logNrs:
            if data_tags.good_data(testDay,logNr):
                if skipCount > 0:
                    print(str(int(comp_count / nr_of_good_logs * 100)), '% completed.   current log:', logNr, '  ', skipCount, 'logs skipped', end='\r')
                    skipCount = 0
                else:
                    print(str(int(comp_count / nr_of_good_logs * 100)), '% completed.   current log:', logNr, end='\r')

                pathLogNr = pathTestDay + '/' + logNr
                raw_data.set_load_path(pathLogNr)
                raw_data.load_raw_data()

                filtered_data = filter_raw_data(raw_data)

                # Remove data that is not required
                filtered_required_data = {}
                for name in filtered_data.get_required_data():
                    try:
                        filtered_required_data[name] = filtered_data.get_data(name)
                    except:
                        raise KeyError('No data with name', name, 'found in filtered_data.')

                filtered_kart_data[logNr] = filtered_required_data

                comp_count += 1

            else:
                skipCount += 1
    return filtered_kart_data

def get_testday_lognr_pairs(pathRootData):
    pairs = []
    testDays = getDirectories(pathRootData)
    testDays.sort()
    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()
        for logNr in logNrs:
            pairs.append([testDay,logNr])
    return pairs