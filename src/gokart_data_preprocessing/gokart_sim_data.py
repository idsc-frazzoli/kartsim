#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:25

@author: mvb
"""
from data_visualization.data_io import dataframe_from_csv, getPKL


class GokartSimData():

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.kartData = {}

        if self.file_path != None:
            self.file_name = self.file_path.split('/')[-1]
            self.load_sim_data()

    def get_data(self,name=None):
        if name == None:
            return self.kartData
        else:
            try:
                return self.kartData[name]['data']
            except:
                raise ('Invalid key. No data found under this model_type.')

    def load_sim_data(self):
        if self.file_name.endswith('.pkl'):
            try:
                dataFrame = getPKL(self.file_path)
                for topic in dataFrame.columns:
                    if 'time' in topic or 'Time' in topic:
                        timeTopic = topic
                        break
                for topic in dataFrame.columns:
                    if topic != timeTopic:
                        self.kartData[topic] = {}
                        self.kartData[topic]['data'] = [dataFrame[timeTopic].values, dataFrame[topic].values]
                        self.kartData[topic]['info'] = [1, 0, 0, 1]
            except:
                print('EmptyDataError: could not read data from file ', self.file_name)
                raise

        elif self.file_name.endswith('.csv'):
            try:
                dataFrame = dataframe_from_csv(self.file_path)
                for topic in dataFrame.columns:
                    if 'time' in topic or 'Time' in topic:
                        timeTopic = topic
                        break
                if len(timeTopic) != 0:
                    for topic in dataFrame.columns:
                        if topic != timeTopic:
                            self.kartData[topic] = {}
                            self.kartData[topic]['data'] = [dataFrame[timeTopic].values, dataFrame[topic].values]
                            self.kartData[topic]['info'] = [1, 0, 0, 1]
                else:
                    print('No column model_type called \'time\' or \'Time\' found in dataset.')
                    raise ValueError
            except:
                print('EmptyDataError: could not read data from file ', self.file_name)
                raise


