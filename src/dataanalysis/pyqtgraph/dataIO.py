#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:25:01 2019

@author: mvb
"""

import pandas as pd
import os


def getCSV(filePath):
    dataFrame = pd.read_csv(str(filePath))
    return dataFrame


def getDirectories(path):
    folders = [f.name for f in os.scandir(path) if f.is_dir()]
    return folders
