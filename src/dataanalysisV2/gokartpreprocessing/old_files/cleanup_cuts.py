#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 05.07.19 11:04

@author: mvb
"""
from dataanalysisV2.data_io import getDirectories, dict_to_csv, dict_from_csv
import os


pathRootData = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts'
testDays = getDirectories(pathRootData)
print(testDays)

lcmfiles = []
i = 0
for testDay in testDays:
    logs = getDirectories(os.path.join(pathRootData, testDay))
    print(logs)
    i = i+len(logs)
    for log in logs:
        for r, d, f in os.walk(os.path.join(pathRootData, testDay, log)):
            for file in f:
                if '.lcm' in file:
                    lcmfiles.append(os.path.join(r, file))

print(lcmfiles)
print(len(lcmfiles))

# for file_path in lcmfiles:
#     os.remove(file_path)
