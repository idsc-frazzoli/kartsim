#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 02.07.19 11:10

@author: mvb
"""
from matplotlib.backends.backend_pdf import PdfPages

import config
from data_visualization.data_io import getPKL
from gokart_data_preprocessing.gokart_sim_data import GokartSimData
import os
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import seaborn as sns
import numpy as np


'''
'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
'pose atheta [rad*s^-2]', 'steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]'
'''

folder_path = '/home/mvb/0_ETH/01_MasterThesis/SimData/20190702-104900_comparison_different_models_closedloop/'
folder_path = '/home/mvb/0_ETH/01_MasterThesis/SimData/20190702-094700_comparison_different_models_1s/'
folder_path = '/home/mvb/0_ETH/01_MasterThesis/SimData/20190702-205704_test_eval_final'
# folder_path = '/home/mvb/0_ETH/01_MasterThesis/SimData/20190702-213054_test_eval_final'

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'w']

x_crop = [0, 200]
# x_crop = [200, 450]

view_plot = True
view_plot = False


topics = [
    ['vehicle lateral acceleration', 'acceleration ' + r'[$\frac{m}{s^2}$]', 'vehicle ay local [m*s^-2]'],
    ['vehicle lateral velocity', 'velocity ' + r'[$\frac{m}{s}$]', 'vehicle vy [m*s^-1]'],
    ['vehicle longitudinal acceleration', 'acceleration ' + r'[$\frac{m}{s^2}$]', 'vehicle ax local [m*s^-2]'],
    ['vehicle longitudinal velocity', 'velocity ' + r'[$\frac{m}{s}$]', 'vehicle vx [m*s^-1]'],
    # ['vehicle rotational acceleration', 'acceleration ' + r'[$\frac{rad}{s^2}$]', 'pose atheta [rad*s^-2]'],
    # ['vehicle rotational velocity', 'velocity ' + r'[$\frac{rad}{s}$]', 'pose vtheta [rad*s^-1]'],
    ]

plot_data = []

for title, unit, topic in topics:
    # plot_this = [
    #     ['20190603T114129_05_sampledlogdata.pkl', topic, 'log data'],
    #     # ['20190603T114129_05_mpc_dynamic_closedloop.csv', topic, 'dynamic MPC model'],
    #     # ['20190603T114129_05_5x64_relu_reg0p0_closedloop.csv', topic, 'learned model'],
    #     ['20190603T114129_05_mpc_dynamic_openloop.csv', topic, 'dynamic MPC model'],
    #     ['20190603T114129_05_5x64_relu_reg0p0_openloop.csv', topic, 'learned model'],
    # ]

    plot_this = [
        ['20190530T160230_03_sampledlogdata.pkl', topic, 'reference'],
        # ['20190530T160230_03_mpc_dynamic_closedloop.csv', topic, 'dynamic MPC model'],
        # ['20190530T160230_03_5x64_relu_reg0p0_closedloop.csv', topic, 'learned model'],
        # ['20190530T160230_03_mpc_dynamic_closedloopinterval.csv', topic, 'dynamic MPC model'],
        # ['20190530T160230_03_5x64_relu_reg0p0_closedloopinterval.csv', topic, 'learned model'],
        ['20190530T160230_03_mpc_dynamic_openloop.csv', topic, 'dynamic MPC model'],
        ['20190530T160230_03_5x64_relu_reg0p0_openloop.csv', topic, 'learned model'],
    ]

    plot_topic = []
    for file_name, topic, name in plot_this:
        sim_data = GokartSimData(file_path=os.path.join(folder_path, file_name))
        vx = sim_data.get_data(topic)
        plot_topic.append([vx[0], vx[1], name])
    plot_data.append([title, unit, plot_topic])

if view_plot:
    # Switch to using white background and black foreground
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    QtGui.QApplication.setGraphicsSystem('raster')
    app = QtGui.QApplication([])
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)

    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1600,800)
    win.setWindowTitle('pyqtgraph example: Plotting')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    plots = {}
    for title, unit, plot_topic in plot_data:
        win.nextRow()
        plots[title + '1'] = win.addPlot(title="Region Selection")

        for i, [x, y, name] in enumerate(plot_topic):
            plots[title + '1'].plot(x, y, pen=colors[i], name=name)
        plots[title + 'lr'] = pg.LinearRegionItem([0,10])
        plots[title + 'lr'].setZValue(-10)
        plots[title + '1'].addItem(plots[title + 'lr'])
        plots[title + '1'].addLegend()
        plots[title + '1'].showGrid(x=True,y=True, alpha=0.3)

        plots[title + '2'] = win.addPlot(title=title)
        plots[title + '2'].addLegend()
        for i, [x, y, name] in enumerate(plot_topic):
            plots[title + '2'].plot(x, y, pen=colors[i], name=name)
        def updatePlot():
            plots[title + '2'].setXRange(*plots[title + 'lr'].getRegion(), padding=0)
        def updateRegion():
            plots[title + 'lr'].setRegion(plots[title + '2'].getViewBox().viewRange()[0])

        plots[title + 'updatePlot'] = updatePlot
        plots[title + 'updateRegion'] = updateRegion
        plots[title + 'lr'].sigRegionChanged.connect(plots[title + 'updatePlot'])
        plots[title + '2'].sigXRangeChanged.connect(plots[title + 'updateRegion'])
        plots[title + '2'].showGrid(x=True,y=True, alpha=0.3)
        updatePlot()

def main():
    for title, unit, plot_topic in plot_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        for i, [x, y, name] in enumerate(plot_topic):
            x_lim = x[x_crop[0]: x_crop[1]]
            y_lim = y[x_crop[0]: x_crop[1]]
            ax.plot(x_lim, y_lim, c=colors[i], label=name)
        ax.set_title(title)
        ax.legend()     #loc='upper left')
        ax.set_ylabel(unit)
        ax.set_xlabel('time ' + r'[$s$]')
        # ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
        ax.autoscale(axis='y')
        fig.tight_layout()

def training_data():
    random_state = 42

    # path_data_set = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets/LearnedModel/20190625-200000_TF_filtered_vel/disturbance.pkl'
    # dataframe = getPKL(path_data_set)
    # dataframe.pop('time [s]')
    #
    # # Split data_set into training and test set
    # train_dataset = dataframe.sample(frac=0.8, random_state=random_state)
    # train_dataset = train_dataset.reset_index(drop=True)
    #
    # test_dataset = dataframe.drop(train_dataset.index)
    # test_dataset = test_dataset.reset_index(drop=True)
    #
    # train_labels_old = train_dataset[['disturbance vehicle ax local [m*s^-2]',
    #                               'disturbance vehicle ay local [m*s^-2]',
    #                               'disturbance pose atheta [rad*s^-2]']]
    # train_features_old = train_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
    #                                 'steer position cal [n.a.]', 'brake position effective [m]',
    #                                 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]
    #
    # test_labels_old = test_dataset[['disturbance vehicle ax local [m*s^-2]',
    #                             'disturbance vehicle ay local [m*s^-2]',
    #                             'disturbance pose atheta [rad*s^-2]']]
    # test_features_old = test_dataset[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
    #                               'steer position cal [n.a.]', 'brake position effective [m]',
    #                               'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]

    path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190717-100934_trustworty_data')
    train_features_trustworthy = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_features_trustworthy = train_features_trustworthy[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                    'steer position cal [n.a.]', 'brake position effective [m]',
                                    'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']] # get rid of time values
    train_labels_trustworthy = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    train_labels_trustworthy = train_labels_trustworthy[['disturbance vehicle ax local [m*s^-2]',
                                  'disturbance vehicle ay local [m*s^-2]',
                                  'disturbance pose atheta [rad*s^-2]']]
    test_features_trustworthy = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_features_trustworthy = test_features_trustworthy[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'steer position cal [n.a.]', 'brake position effective [m]',
                                     'motor torque cmd left [A_rms]',
                                     'motor torque cmd right [A_rms]']]  # get rid of time values
    test_labels_trustworthy = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))
    test_labels_trustworthy = test_labels_trustworthy[['disturbance vehicle ax local [m*s^-2]',
                                 'disturbance vehicle ay local [m*s^-2]',
                                 'disturbance pose atheta [rad*s^-2]']]

    path_data_set = os.path.join(config.directories['root'], 'Data/MLPDatasets/20190717-211215_high_slip_angles')
    train_features_high_slip = getPKL(os.path.join(path_data_set, 'train_features.pkl'))
    train_features_high_slip = train_features_high_slip[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                             'steer position cal [n.a.]', 'brake position effective [m]',
                                             'motor torque cmd left [A_rms]',
                                             'motor torque cmd right [A_rms]']]  # get rid of time values
    train_labels_high_slip = getPKL(os.path.join(path_data_set, 'train_labels.pkl'))
    train_labels_high_slip = train_labels_high_slip[['disturbance vehicle ax local [m*s^-2]',
                                         'disturbance vehicle ay local [m*s^-2]',
                                         'disturbance pose atheta [rad*s^-2]']]
    test_features_high_slip = getPKL(os.path.join(path_data_set, 'test_features.pkl'))
    test_features_high_slip = test_features_high_slip[['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                           'steer position cal [n.a.]', 'brake position effective [m]',
                                           'motor torque cmd left [A_rms]',
                                           'motor torque cmd right [A_rms]']]  # get rid of time values
    test_labels_high_slip = getPKL(os.path.join(path_data_set, 'test_labels.pkl'))
    test_labels_high_slip = test_labels_high_slip[['disturbance vehicle ax local [m*s^-2]',
                                       'disturbance vehicle ay local [m*s^-2]',
                                       'disturbance pose atheta [rad*s^-2]']]
    plot_this = [
        [train_features_trustworthy, train_features_high_slip, ],
        # [test_features_trustworthy, test_features_high_slip, ],
    ]

    i=1
    topics = ['vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                     'steer position cal [n.a.]', 'brake position effective [m]',
                                     'motor torque cmd left [A_rms]',
                                     'motor torque cmd right [A_rms]']
    pdffilepath = '/home/mvb/0_ETH/01_MasterThesis/kartsim_files/Evaluation/' + '_data_density_of_observations.pdf'
    pdf = PdfPages(pdffilepath)
    for topic in topics:
        plt.figure(i)
        for num, data_pair in enumerate(plot_this):
            # plt.subplot(2, 1, num + 1)
            for data in data_pair:
                sns.distplot(data[topic])
        plt.ylabel('density')
        plt.legend(['fast and slow driving', 'only fast driving'])
        i+=1
        pdf.savefig()
        plt.close()

    pdf.close()



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if view_plot:
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    else:
        # evaluate()
        training_data()