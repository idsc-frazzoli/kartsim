#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:56:56 2019

@author: mvb
"""

#import matlab.engine
import matlab.engine
eng = matlab.engine.find_matlab()

eng = matlab.engine.connect_matlab(name=None)

print(eng)

