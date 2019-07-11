#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:18

@author: mvb
"""
import numpy as np
import time

from data_visualization.data_io import getDirectories, dict_to_csv, dict_from_csv
from gokart_data_preprocessing.preprocessing import filter_raw_data
# from data_visualization.gokart_data_preprocessing.importdata import setListItems
from gokart_data_preprocessing.gokart_raw_data import GokartRawData

class TagRawData:

    def __init__(self, pathRootData=None):
        if pathRootData == None:
            raise FileNotFoundError(
                'Path of raw data undefined. Please specify where the raw Gokart log data is located.')

        self.pathRootData = pathRootData
        self._preproParamsFileName = 'preproParams'
        self._preproParamsFilePath = pathRootData + '/' + self._preproParamsFileName + '.csv'

        self.path_logs = ()
        self.total_number_of_logs = 0
        self.good_number_of_logs = 0

        try:
            self.data_tags = dict_from_csv(self._preproParamsFilePath)
            print('Parameter file for preprocessing located and loaded.')
        except:
            print('Parameter file for preprocessing does not exist. Creating file...')
            self.data_tags = {}

        # self.initialize_parameters()

    def good_data(self, test_day, log_nr):
        return int(self.data_tags[test_day][log_nr]['goodData'])

    def initialize_parameters(self, overwrite=True):
        self.path_logs = ()
        self.total_number_of_logs = 0
        testDays = getDirectories(self.pathRootData)
        testDays.sort()

        if overwrite:
            self.data_tags = {}

        for testDay in testDays:
            if testDay not in self.data_tags:
                self.data_tags[testDay] = {}
            pathTestDay = self.pathRootData + '/' + testDay
            logNrs = getDirectories(pathTestDay)
            logNrs.sort()
            self.total_number_of_logs += len(logNrs)

            for logNr in logNrs:
                if logNr in self.data_tags[testDay]:
                    print(logNr, ' already done. Continuing with next logNr')
                    continue
                else:
                    self.data_tags[testDay][logNr] = {}
                    self.data_tags[testDay][logNr]['trustworthy data'] = 1
                    self.path_logs = self.path_logs + ((testDay, logNr, pathTestDay + '/' + logNr),)
        self.good_number_of_logs = self.total_number_of_logs

    def sort_out_data(self, required_data_list, required_tags_list, exclusion_tags_list):
        self.good_number_of_logs = 0
        for day in self.data_tags:
            for log in self.data_tags[day]:
                self.good_number_of_logs += 1
                if int(self.data_tags[day][log]['trustworthy data']) == 0:
                    self.data_tags[day][log]['goodData'] = 0
                    self.good_number_of_logs -= 1
                    continue
                self.data_tags[day][log]['goodData'] = 1
                for topic in self.data_tags[day][log]:
                    if (topic in required_tags_list and int(self.data_tags[day][log][topic]) == 0) or \
                            (topic in required_data_list and int(self.data_tags[day][log][topic]) == 0) or \
                            (topic in exclusion_tags_list and int(self.data_tags[day][log][topic]) == 1):
                        self.data_tags[day][log]['goodData'] = 0
                        self.good_number_of_logs -= 1
                        break

    def save_prepro_params(self):
        dict_to_csv(self._preproParamsFilePath, self.data_tags)
        print('preproParams saved to ', self._preproParamsFilePath)

    def get_tags(self):
        return self.data_tags

    def tag_log_files(self, overwrite):
        t = time.time()

        self.initialize_parameters(overwrite)

        comp_count = 0
        for testDay, logNr, path in self.path_logs:

            raw_data = GokartRawData(path)
            # kartData, allDataNames = setListItems(path)
            vmuDataNames = ['pose x [m]',
                            'pose y [m]',
                            'pose theta [rad]',
                            'pose vtheta [rad*s^-1]',
                            'vehicle vy [m*s^-1]',
                            'pose atheta [rad*s^-2]',
                            'vmu ax [m*s^-2]',
                            'vmu ay [m*s^-2]',
                            'vehicle ax local [m*s^-2]',
                            'vehicle ay local [m*s^-2]',]
            filtered_data = filter_raw_data(raw_data, vmuDataNames)
            kartData = filtered_data.get_data()

            statusInfo = logNr + ':  '



            # ___pose quality
            bad_pose_qual_count = sum(i < 0.5 for i in kartData['pose quality [n.a.]']['data'][1])
            bad_pose_qual_ratio = bad_pose_qual_count/len(kartData['pose quality [n.a.]']['data'][1])
            if bad_pose_qual_count > 100 or bad_pose_qual_ratio > 0.25:
                self.data_tags[testDay][logNr]['pose quality'] = 0
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                statusInfo = statusInfo + 'pose quality insufficient,  '
            else:
                self.data_tags[testDay][logNr]['pose quality'] = 1

            # ___distance
            dx = np.subtract(kartData['pose x [m]']['data'][1][1:], kartData['pose x [m]']['data'][1][:-1])
            dy = np.subtract(kartData['pose y [m]']['data'][1][1:], kartData['pose y [m]']['data'][1][:-1])
            dist = np.sum(np.sqrt(np.square(dx) + np.square(dy)))
            if dist > 100:
                self.data_tags[testDay][logNr]['multiple laps'] = 1
            else:
                self.data_tags[testDay][logNr]['multiple laps'] = 0

            # ___driving style
            if np.std(kartData['vehicle vy [m*s^-1]']['data'][1]) > 0.2:
                self.data_tags[testDay][logNr]['high slip angles'] = 1
            else:
                self.data_tags[testDay][logNr]['high slip angles'] = 0

            # ___reverse
            if np.min(kartData['vehicle vx [m*s^-1]']['data'][1]) < -0.2:
                self.data_tags[testDay][logNr]['reverse'] = 1
            else:
                self.data_tags[testDay][logNr]['reverse'] = 0

            # ___vehicle ax local
            if np.max(kartData['vehicle ax local [m*s^-2]']['data'][1]) > 2.5 or np.min(
                    kartData['vehicle ax local [m*s^-2]']['data'][1]) < -14.0:
                self.data_tags[testDay][logNr]['vehicle ax local [m*s^-2]'] = 0
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                statusInfo = statusInfo + 'unrealistic vehicle ax,  '
            else:
                self.data_tags[testDay][logNr]['vehicle ax local [m*s^-2]'] = 1

            # ___vehicle ay local
            if np.max(kartData['vehicle ay local [m*s^-2]']['data'][1]) > 9.0 or np.min(
                    kartData['vehicle ay local [m*s^-2]']['data'][1]) < -9.0:
                self.data_tags[testDay][logNr]['vehicle ay local [m*s^-2]'] = 0
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                statusInfo = statusInfo + 'unrealistic vehicle ay,  '
            else:
                self.data_tags[testDay][logNr]['vehicle ay local [m*s^-2]'] = 1

            # ___pose atheta
            if np.max(kartData['pose atheta [rad*s^-2]']['data'][1]) > 12.0 or np.min(
                    kartData['pose atheta [rad*s^-2]']['data'][1]) < -12.0:
                self.data_tags[testDay][logNr]['pose atheta [rad*s^-2]'] = 0
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                statusInfo = statusInfo + 'unrealistic pose atheta,  '
            else:
                self.data_tags[testDay][logNr]['pose atheta [rad*s^-2]'] = 1

            #___x velocity
            if np.max(kartData['vehicle vx [m*s^-1]']['data'][1]) > 12.0:
                statusInfo = statusInfo + 'unrealistic vehicle vx,  '
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                self.data_tags[testDay][logNr]['vehicle vx [m*s^-1]'] = 0
            else:
                self.data_tags[testDay][logNr]['vehicle vx [m*s^-1]'] = 1

            # ___y velocity
            if np.max(kartData['vehicle vy [m*s^-1]']['data'][1]) > 12 or np.min(
                    kartData['vehicle vy [m*s^-1]']['data'][1]) < -12:
                statusInfo = statusInfo + 'unrealistic vehicle vy,  '
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                self.data_tags[testDay][logNr]['vehicle vy [m*s^-1]'] = 0
            else:
                self.data_tags[testDay][logNr]['vehicle vy [m*s^-1]'] = 1

            # ___theta velocity
            if np.max(kartData['pose vtheta [rad*s^-1]']['data'][1]) > 4.0 or np.min(
                    kartData['pose vtheta [rad*s^-1]']['data'][1]) < -4.0:
                statusInfo = statusInfo + 'unrealistic pose vtheta,  '
                self.data_tags[testDay][logNr]['trustworthy data'] = 0
                self.data_tags[testDay][logNr]['pose vtheta [rad*s^-1]'] = 0
            else:
                self.data_tags[testDay][logNr]['pose vtheta [rad*s^-1]'] = 1

            # ___steering
            lenSteerCmd = len(kartData['steer torque cmd [n.a.]']['data'][1])
            if lenSteerCmd == 1:
                statusInfo = statusInfo + 'steering cmd data missing,  '
                self.data_tags[testDay][logNr]['steer torque cmd [n.a.]'] = 0
            elif kartData['steer torque cmd [n.a.]']['data'][1][int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)].count(
                    0) / len(
                kartData['steer torque cmd [n.a.]']['data'][1][
                int(lenSteerCmd / 10):int(lenSteerCmd / 10 * 9)]) > 0.05:
                statusInfo = statusInfo + 'steering cmd data: too many zeros...,  '
                self.data_tags[testDay][logNr]['steer torque cmd [n.a.]'] = 0
            elif np.abs(np.mean(kartData['steer torque cmd [n.a.]']['data'][1])) < 0.01 and np.std(
                    kartData['steer torque cmd [n.a.]']['data'][1]) < 0.1:
                statusInfo = statusInfo + 'steering cmd data insufficient,  '
                self.data_tags[testDay][logNr]['steer torque cmd [n.a.]'] = 0
            else:
                self.data_tags[testDay][logNr]['steer torque cmd [n.a.]'] = 1

            lenSteerPos = len(kartData['steer position cal [n.a.]']['data'][1])
            if lenSteerPos == 1:
                statusInfo = statusInfo + 'steering pos cal data missing,  '
                self.data_tags[testDay][logNr]['steer position cal [n.a.]'] = 0
            elif kartData['steer position cal [n.a.]']['data'][1][
                 int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)].count(
                0) / len(
                kartData['steer position cal [n.a.]']['data'][1][
                int(lenSteerPos / 10):int(lenSteerPos / 10 * 9)]) > 0.05:
                statusInfo = statusInfo + 'steering pos cal data: too many zeros...,  '
                self.data_tags[testDay][logNr]['steer position cal [n.a.]'] = 0
            elif np.abs(np.mean(kartData['steer position cal [n.a.]']['data'][1])) < 0.01 and np.std(
                    kartData['steer position cal [n.a.]']['data'][1]) < 0.1:
                statusInfo = statusInfo + 'steering pos cal data missing or insufficient,  '
                self.data_tags[testDay][logNr]['steer position cal [n.a.]'] = 0
            else:
                self.data_tags[testDay][logNr]['steer position cal [n.a.]'] = 1

            # ___brake
            if np.max(kartData['brake position cmd [m]']['data'][1]) < 0.025 and np.mean(
                    kartData['brake position cmd [m]']['data'][1]) < 0.004:
                statusInfo = statusInfo + 'brake position cmd [m] data missing or insufficient,  '
                self.data_tags[testDay][logNr]['brake position cmd [m]'] = 0
            else:
                self.data_tags[testDay][logNr]['brake position cmd [m]'] = 1

            if np.max(kartData['brake position effective [m]']['data'][1]) < 0.025 and np.mean(
                    kartData['brake position effective [m]']['data'][1]) < 0.004:
                statusInfo = statusInfo + 'brake position effective [m] data missing or insufficient,  '
                self.data_tags[testDay][logNr]['brake position effective [m]'] = 0
            else:
                self.data_tags[testDay][logNr]['brake position effective [m]'] = 1

            # ___VMU
            if np.abs(np.mean(kartData['vmu ax [m*s^-2]']['data'][1])) < 0.01 and np.std(
                    kartData['vmu ax [m*s^-2]']['data'][1]) < 0.01:
                statusInfo = statusInfo + 'vmu ax [m*s^-2] data missing or insufficient,  '
                self.data_tags[testDay][logNr]['vmu ax [m*s^-2]'] = 0
            else:
                self.data_tags[testDay][logNr]['vmu ax [m*s^-2]'] = 1

            if np.abs(np.mean(kartData['vmu ay [m*s^-2]']['data'][1])) < 0.01 and np.std(
                    kartData['vmu ay [m*s^-2]']['data'][1]) < 0.05:
                statusInfo = statusInfo + 'vmu ay [m*s^-2] data missing or insufficient,  '
                self.data_tags[testDay][logNr]['vmu ay [m*s^-2]'] = 0
            else:
                self.data_tags[testDay][logNr]['vmu ay [m*s^-2]'] = 1

            if np.abs(np.mean(kartData['vmu vtheta [rad*s^-1]']['data'][1])) < 0.01 and np.std(
                    kartData['vmu vtheta [rad*s^-1]']['data'][1]) < 0.05:
                statusInfo = statusInfo + 'vmu vtheta [rad*s^-1] data missing or insufficient,  '
                self.data_tags[testDay][logNr]['vmu vtheta [rad*s^-1]'] = 0
            else:
                self.data_tags[testDay][logNr]['vmu vtheta [rad*s^-1]'] = 1

            # ___MH model specific
            if self.data_tags[testDay][logNr]['brake position effective [m]']:
                self.data_tags[testDay][logNr]['MH AB [m*s^-2]'] = 1
                self.data_tags[testDay][logNr]['MH TV [rad*s^-2]'] = 1
            else:
                self.data_tags[testDay][logNr]['MH AB [m*s^-2]'] = 0
                self.data_tags[testDay][logNr]['MH TV [rad*s^-2]'] = 0

            if self.data_tags[testDay][logNr]['steer position cal [n.a.]']:
                self.data_tags[testDay][logNr]['MH BETA [rad]'] = 1
            else:
                self.data_tags[testDay][logNr]['MH BETA [rad]'] = 0

            statusInfo = statusInfo + 'done'
            comp_count += 1
            print(statusInfo)
            print(str(int(comp_count / self.total_number_of_logs * 100)), '% completed.  elapsed time:',
                  int(time.time() - t), "s", end='\r')
