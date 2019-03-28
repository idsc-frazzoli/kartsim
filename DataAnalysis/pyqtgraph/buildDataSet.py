#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""


self.pathRootData = '/home/mvb/0_ETH/MasterThesis/Logs_GoKart/LogData/dynamics'
testDays = self.getDirectories(self.pathRootData)
testDays.sort()
defaultDay = testDays[-1]
self.pathTestDay = self.pathRootData + '/' + defaultDay
LogNrs = self.getDirectories(self.pathTestDay)
LogNrs.sort()
defaultLogNr = LogNrs[21]
self.pathLogNr = self.pathTestDay + '/' + defaultLogNr
        
def getDirectories(self, path):
    folders = [f.name for f in os.scandir(path) if f.is_dir()]
    return folders