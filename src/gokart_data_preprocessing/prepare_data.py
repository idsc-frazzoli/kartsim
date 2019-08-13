#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
from data_visualization.data_io import getDirectories
from gokart_data_preprocessing.preprocessing import filter_raw_data

from gokart_data_preprocessing.gokart_raw_data import GokartRawData


def prepare_dataset(pathRootData, data_tags, required_data_list, start_from=None):
    test_log_pairs = get_testday_lognr_pairs(pathRootData, start_from)
    total_nr_of_logs = len(test_log_pairs)
    good_logs_list = [data_tags.good_data(day, log) for day, log in test_log_pairs]
    nr_of_good_logs = sum(good_logs_list)

    print(nr_of_good_logs, 'of', total_nr_of_logs, 'logs contain the necessary tags.')

    filtered_kart_data = {}
    skipCount = 0
    comp_count = 0

    raw_data = GokartRawData(required_data_list=required_data_list)

    for testDay, logNr in test_log_pairs:
        if data_tags.good_data(testDay,logNr):
            if skipCount > 0:
                print(str(int(comp_count / nr_of_good_logs * 100)), '% completed.   current log:', logNr, '  ', skipCount, 'logs skipped', end='\r')
                skipCount = 0
            else:
                print(str(int(comp_count / nr_of_good_logs * 100)), '% completed.   current log:', logNr, end='\r')

            pathLogNr = pathRootData + '/' + testDay + '/' + logNr
            raw_data.set_load_path(pathLogNr)
            raw_data.load_raw_data()

            filtered_data = filter_raw_data(raw_data)

            # Remove data that is not required
            filtered_required_data = {}
            for name in filtered_data.get_required_data():
                try:
                    filtered_required_data[name] = filtered_data.get_data(name)
                except:
                    raise KeyError('No data with model_type', name, 'found in filtered_data.')

            filtered_kart_data[logNr] = filtered_required_data

            comp_count += 1

        else:
            skipCount += 1
    return filtered_kart_data

def get_testday_lognr_pairs(pathRootData, start_from):
    pairs = []
    testDays = getDirectories(pathRootData)
    testDays.sort()

    if start_from != None:
        index = testDays.index(start_from)
        testDays = testDays[index:]

    for testDay in testDays:
        pathTestDay = pathRootData + '/' + testDay
        logNrs = getDirectories(pathTestDay)
        logNrs.sort()
        for logNr in logNrs:
            pairs.append([testDay,logNr])
    return pairs