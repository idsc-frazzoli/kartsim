#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:25

@author: mvb
"""
import os
import numpy as np
from dataanalysisV2.data_io import dataframe_from_csv
from dataanalysisV2.mathfunction import interpolation
import time
from copy import copy

class GokartRawData:
    def __init__(self, load_path=None, required_data_list=None):
        self.load_path = load_path
        self.kartData = {}
        self.required_data_list = required_data_list

        if self.load_path != None:
            self.log_nr = self.load_path.split('/')[-1]
            self.load_raw_data()

    def get_key_names(self):
        return list(self.kartData.keys())

    def get_required_data(self):
        return self.required_data_list

    def get_data(self,name=None):
        if name == None:
            return self.kartData
        else:
            try:
                return self.kartData[name]['data']
            except:
                raise ('Invalid key. No data found under this name.')

    def set_data(self,name, x=[], y=[]):
        if len(list(x)) > 0 and len(list(y)) > 0:
            self.kartData[name]['data'][0] = x
            self.kartData[name]['data'][1] = y
        elif len(list(y)) > 0:
            self.kartData[name]['data'][1] = y
        elif len(list(x)) > 0:
            self.kartData[name]['data'][0] = x
        else:
            raise ValueError('Please specify data to be set.')

    def set_load_path(self, new_load_path):
        self.load_path = new_load_path

    def set_required_data_list(self, required_tags_list):
        self.required_data_list = required_tags_list

    def get_attribute(self, data_name, attribute):
        if attribute == 'sigma':
            index = 1
        elif attribute == 'width':
            index = 2
        else:
            raise ValueError('Attribute name not found. Choose from ["sigma","width"]')

        return self.kartData[data_name]['info'][index]

    def get_dependencies(self, key_list):
        dependencies = []
        for key in key_list:
            if key in list(self.kartData.keys()):
                dependencies += self.kartData[key]['info'][0]
                dependencies += self.get_dependencies(self.kartData[key]['info'][0])
        return dependencies

    def load_raw_data(self):
        self.kartData = {}
        files = []
        groups = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.load_path):
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(r, file))

        # Add Preprocessed Data
        groups.append(['pose x [m]', ['pose x atvmu [m]'], True, 5, 50, 1, 1])
        groups.append(['pose y [m]', ['pose y atvmu [m]'], True, 5, 50, 1, 1])
        groups.append(['pose vx [m*s^-1]', ['pose x [m]'], True, 5, 50, 1, 1])
        groups.append(['pose vy [m*s^-1]', ['pose y [m]'], True, 5, 50, 1, 1])
        groups.append(['pose vtheta [rad*s^-1]', ['pose theta [rad]'], True, 5, 50, 1, 1])
        groups.append(['vehicle vy [m*s^-1]', ['vehicle vy atvmu [m*s^-1]', 'pose vtheta [rad*s^-1]'], True, 0, 0, 2, 1])
        groups.append(['vehicle ax local [m*s^-2]', ['vehicle vx [m*s^-1]'], True, 0, 0, 1, 1])
        groups.append(['vehicle ay local [m*s^-2]', ['vehicle vy [m*s^-1]'], True, 0, 0, 1, 1])
        groups.append(['pose ax [m*s^-2]', ['pose vx [m*s^-1]'], True, 20, 200, 2, 1])
        groups.append(['pose ay [m*s^-2]', ['pose vy [m*s^-1]'], True, 20, 200, 2, 1])
        groups.append(['pose atheta [rad*s^-2]', ['pose vtheta [rad*s^-1]'], True, 0, 0, 2, 1])
        groups.append(['vehicle slip angle [rad]', ['pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]'], True, 0, 0, 2, 1])
        groups.append(['vmu ax [m*s^-2]', ['pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]'], True, 0, 0, 3, 1])
        groups.append(['vmu ay [m*s^-2]', ['pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]'], True, 0, 0, 3, 1])
        groups.append(['vehicle ax total [m*s^-2]',
                       ['pose theta [rad]', 'pose vtheta [rad*s^-1]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]',
                        'vehicle vy [m*s^-1]'],
                       True, 0, 0, 3, 1])
        groups.append(['vehicle ay total [m*s^-2]',
                       ['pose theta [rad]', 'pose vtheta [rad*s^-1]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]',
                        'vehicle vy [m*s^-1]'],
                       True, 0, 0, 3, 1])
        groups.append(['vehicle ax only transl [m*s^-2]',
                       ['pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'pose ax [m*s^-2]', 'pose ay [m*s^-2]'],
                       True, 0, 0, 3, 1])
        groups.append(['vehicle ay only transl [m*s^-2]',
                       ['pose theta [rad]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'pose ax [m*s^-2]', 'pose ay [m*s^-2]'],
                       True, 0, 0, 3, 1])
        groups.append(['MH power accel rimo left [m*s^-2]',
                       ['motor torque cmd left [A_rms]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]'],
                       True, 0, 0, 4, 1])
        groups.append(['MH power accel rimo right [m*s^-2]',
                       ['motor torque cmd right [A_rms]', 'pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]'],
                       True, 0, 0, 4, 1])
        groups.append(['MH AB [m*s^-2]',
                       ['pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'brake position effective [m]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]', 'MH power accel rimo left [m*s^-2]',
                        'MH power accel rimo right [m*s^-2]'],
                       True, 0, 0, 5, 1])
        groups.append(['MH TV [rad*s^-2]',
                       ['pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'vehicle slip angle [rad]', 'vehicle vx [m*s^-1]', 'MH power accel rimo left [m*s^-2]',
                        'MH power accel rimo right [m*s^-2]'],
                       True, 0, 0, 5, 1])
        groups.append(['MH BETA [rad]',
                       ['steer position cal [n.a.]'],
                       True, 0, 0, 1, 1])

        for name, dependencies, vis, sig, wid, order, scale in groups:
            self.kartData[name] = {}
            self.kartData[name]['data'] = [[], []]
            # item.info = [visible, filter_sigma, filter_width, order, scale]
            self.kartData[name]['info'] = [dependencies, sig, wid]

        groups = []
        for name in files:
            if 'pose.lidar' in name:
                groups.append(['pose x atvmu [m]', 0, 1, name, True, 0, 0, 0, 1])
                groups.append(['pose y atvmu [m]', 0, 2, name, True, 0, 0, 0, 1])
                groups.append(['pose theta [rad]', 0, 3, name, True, 5, 50, 0, 1])
                groups.append(['vehicle vx [m*s^-1]', 0, 5, name, True, 10, 100, 0, 1])
                groups.append(['vehicle vy atvmu [m*s^-1]', 0, 6, name, True, 10, 100, 0, 1])
            elif 'steer.put' in name:
                groups.append(['steer torque cmd [n.a.]', 0, 2, name, True, 0, 0, 0, 1])
            elif 'steer.get' in name:
                groups.append(['steer torque eff [n.a.]', 0, 5, name, True, 0, 0, 0, 1])
                groups.append(['steer position raw [n.a.]', 0, 8, name, True, 0, 0, 0, 1])
            elif 'status.get' in name:
                groups.append(['steer position cal [n.a.]', 0, 1, name, True, 0, 0, 0, 1])
            elif 'linmot.put' in name:
                groups.append(['brake position cmd [m]', 0, 1, name, True, 0, 0, 0, 1])
            elif 'linmot.get' in name:
                groups.append(['brake position effective [m]', 0, 1, name, True, 0, 0, 0, 1])
            elif 'rimo.put' in name:
                groups.append(['motor torque cmd left [A_rms]', 0, 1, name, True, 0, 0, 0, 1])
                groups.append(['motor torque cmd right [A_rms]', 0, 2, name, True, 0, 0, 0, 1])
            elif 'rimo.get' in name:
                groups.append(['motor rot rate left [rad*s^-1]', 0, 2, name, True, 0, 0, 0, 1])
                groups.append(['motor rot rate right [rad*s^-1]', 0, 9, name, True, 0, 0, 0, 1])
            elif 'vmu931' in name:
                groups.append(['vmu ax atvmu (forward) [m*s^-2]', 0, 2, name, True, 70, 700, 0, 1])
                groups.append(['vmu ay atvmu (left)[m*s^-2]', 0, 3, name, True, 70, 700, 0, 1])
                groups.append(['vmu vtheta [rad*s^-1]', 0, 4, name, True, 5, 50, 0, 1])
        groups.sort()

        if self.required_data_list != None:
            required_list = copy(self.required_data_list)
            required_list = required_list + self.get_dependencies(required_list)
            required_list.reverse()

            unique_list = []
            for element in required_list:
                if element not in unique_list:
                    unique_list.append(element)

            required_list = unique_list
            name_list = required_list

        else:
            name_list = list(np.array(groups)[:,0])

        for name, timeIndex, dataIndex, fileName, vis, sig, wid, order, scale in groups:
            if name in name_list:
                try:
                    dataFrame = dataframe_from_csv(fileName)

                    xRaw = dataFrame.iloc[:, timeIndex]
                    yRaw = dataFrame.iloc[:, dataIndex]

                    if name == 'vmu vtheta':
                        if int(self.log_nr[5:8]) > 509:
                            yRaw = -yRaw

                    if name == 'pose theta [rad]':
                        dy = np.abs(np.subtract(np.array(yRaw[1:]), np.array(yRaw[:-1])))
                        indices = np.where(dy > 1)
                        for index in indices[0]:
                            yRaw[index + 1:] = yRaw[index + 1:] - np.sign((yRaw[index + 1] - yRaw[index])) * 2 * np.pi

                    if name in ['vmu ax atvmu (forward) [m*s^-2]', 'vmu ay atvmu (left)[m*s^-2]', 'vmu vtheta [rad*s^-1]']:
                        xRaw, yRaw = interpolation(xRaw, yRaw, xRaw.iloc[0], xRaw.iloc[-1], 0.001)

                except:
                    print('EmptyDataError for ', name, ': could not read data from file ', fileName)
                    xRaw, yRaw = [0], [0]


                self.kartData[name] = {}
                self.kartData[name]['data'] = [list(xRaw), list(yRaw)]  # item.data = [x_data, y_data]
                self.kartData[name]['info'] = [[],sig, wid]  # item.info = [visible,
                # filter_sigma, filter_width, order, scale]

        if self.required_data_list != None:
            self.kartData = dict([(key,self.kartData[key]) for key in required_list if key in self.kartData.keys() ])

        if len(groups) == 18:
            pass
        else:
            print('ACHTUNG! Missing Data in ', self.load_path)

