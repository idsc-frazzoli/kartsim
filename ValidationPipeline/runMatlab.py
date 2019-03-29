#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

import matlab.engine

print('Looking for shared Matlab session...')
name = matlab.engine.find_matlab()
if len(name) > 0:
    print('Matlab session ' + str(name[0]) + ' started.')
try:
    eng
except NameError: 
    eng = None
    print('No Matlab session connected!')
if eng == None:
    print('Connecting to shared Matlab session...')
    eng = matlab.engine.connect_matlab(name=None, background=True)

if eng == None:
    print('No Matlab session exists! Please start a Matlab session \n')
else:
    print(eng.sqrt(4.0))


print('Done')
