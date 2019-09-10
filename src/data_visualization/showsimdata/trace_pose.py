#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 02.07.19 11:10

@author: mvb
"""
import time

from matplotlib.backends.backend_pdf import PdfPages

import config
from data_visualization.data_io import getPKL, dataframe_from_csv, getDirectories
from gokart_data_preprocessing.gokart_sim_data import GokartSimData
import os
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import seaborn as sns
import numpy as np

from simulator.model.kinematic_mpc_model import KinematicVehicleMPC

'''
'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
'pose atheta [rad*s^-2]', 'steer position cal [n.a.]', 'brake position effective [m]', 'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]'
'''

pathRootSimData = '/home/mvb/0_ETH/01_MasterThesis/SimData/lookatlogs'
# defaultSim = 'circle_right'
# defaultSim = 'figure_of_8_1'
# defaultSim = 'figure_of_8_2'
# defaultSim = 'mpc_circle_left'
# defaultSim = 'mpc_circle_right'
# defaultSim = 'mpc_straight'
# defaultSim = 'jans_logs'
# defaultSim = 'mpc_with_dymmodel_vd_nn_logdata'
# defaultSim = 'kin_test'
# defaultSim = '20190827_mpc_test_driving_logs'
# defaultSim = '20190830_mpc_vs_nn_forreal'
# defaultSim = '20190907_kin_nn_vs_mpc'
# defaultSim = 'test'
# defaultSim = 'test1'
defaultSim = '20190909_mpc_vs_kin_nn_and_no_model'
pathsimdata = pathRootSimData + '/' + defaultSim

simfiles = []

for r, d, f in os.walk(pathsimdata):
    for file in f:
        if '.csv' in file or '.pkl' in file:
            simfiles.append([os.path.join(r, file), file])
simfiles.sort()
colors = ['b', 'r', 'g', 'c', 'm', (200, 200, 0), 'b', 'g', 'r', 'c', 'm', (200, 200, 0)]
plot_this = [
    # 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
    # 'pose atheta [rad*s^-2]'
    # 'vehicle vx [m*s^-1]',
    # 'vehicle vy [m*s^-1]',
    # 'pose vtheta [rad*s^-1]',
    # 'steer position cal [n.a.]',
    # 'pose theta [rad]',
    # 'turning angle [n.a]',
    # 'acceleration rear axle [m*s^-2]',
    'acceleration torque vectoring [rad*s^-2]',
    # 'motor torque cmd left [A_rms]',
    # 'motor torque cmd right [A_rms]',
]
kart_l = 1.19
kart_w = 1.0
wheel_l = 0.2

pose_data = []
# delay = np.array([[0,5.7], [1,6.7], [2,7.5], [3,5.0], [4,7.2], [5,7.2]]) #test_mpc_dyn
delay = np.array([[0,8.4], [1,20.6], [2,7.4], [3,7.3],])
# delay = np.array([[0,0], [1,0], [2,0], [3,0], [4,0], [5,0]])
for index, [file_path, file_name] in enumerate(simfiles):
    if '.csv' in file_path:
        data_frame = dataframe_from_csv(file_path)
    else:
        data_frame = getPKL(file_path)
    if index in delay[:,0]:
        data_frame = data_frame.iloc[int(delay[index,1]*10):,:]
        data_frame = data_frame.reset_index()
        data_frame['time [s]'] = data_frame['time [s]'].values - delay[index,1]
    print(data_frame.head())
    if 'steer position cal [n.a.]' in data_frame.columns:
        data_frame['turning angle [n.a]'], data_frame['acceleration rear axle [m*s^-2]'], data_frame[
            'acceleration torque vectoring [rad*s^-2]'] = KinematicVehicleMPC().transform_inputs(
            data_frame['steer position cal [n.a.]'],
            data_frame['brake position effective [m]'],
            data_frame['motor torque cmd left [A_rms]'],
            data_frame['motor torque cmd right [A_rms]'],
            data_frame['vehicle vx [m*s^-1]'],
        )
    # pose_data.append([file_name, data_frame[
    #     ['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
    #      'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]', 'brake position effective [m]',
    #      'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']]])
    pose_data.append([file_name, data_frame[
        ['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
         'pose vtheta [rad*s^-1]', 'steer position cal [n.a.]', 'brake position effective [m]',
         'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]', 'turning angle [n.a]',
         'acceleration rear axle [m*s^-2]', 'acceleration torque vectoring [rad*s^-2]']]])
dt = data_frame['time [s]'][1] - data_frame['time [s]'][0]
# Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
# mw = QtGui.QMainWindow()
# mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(800, 400)
win.setWindowTitle('Sim Visualization')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

proxy = QtGui.QGraphicsProxyWidget()
button = QtGui.QPushButton('play/pause')
button.setCheckable(True)
proxy.setWidget(button)

p3 = win.addLayout(row=2, col=0)
p3.addItem(proxy, row=1, col=1)

plots = {}
win.nextRow()
plots['1'] = win.addPlot(title="Time Line")
plots['1'].addLegend()

plots['lr'] = pg.InfiniteLine(movable=True, pen='g')
plots['1'].addItem(plots['lr'])
plots['1'].showGrid(x=True, y=True, alpha=0.3)

plots['2'] = win.addPlot(title='Pose')
plots['2'].addLegend()
plots['2'].setAspectLocked(lock=True, ratio=1)
plots['2'].disableAutoRange()
plots['2'].setXRange(23, 53, padding=0, update=True)
plots['2'].setYRange(25, 58, padding=0, update=True)

x_pos_data = []
y_pos_data = []
x_vel_data = []
y_vel_data = []
theta_data = []
beta_data = []

karts = {}
traces = {}
wheels_RL = {}
wheels_RR = {}
wheels_FL = {}
wheels_FR = {}
turn_circle_vel = {}
turn_circle_steer = {}
color_count = 0
t_max = 0
# for index, ([name, data], namename) in enumerate(
#         zip(pose_data, ['kinematic model + NN', 'dynamic model + NN', 'reference'])):
for index, (name, data) in enumerate(pose_data):
    if data['time [s]'].values[-1] > t_max:
        t_max = data['time [s]'].values[-1]
    x_pos_data.append(data['pose x [m]'].values)
    y_pos_data.append(data['pose y [m]'].values)
    theta_data.append(data['pose theta [rad]'].values)
    # beta_data.append(data['steer position cal [n.a.]'].values)
    beta_data.append(data['turning angle [n.a]'].values)
    x_vel_data.append(data['vehicle vx [m*s^-1]'].values)
    y_vel_data.append(data['vehicle vy [m*s^-1]'].values)

    karts[str(index)] = pg.QtGui.QGraphicsRectItem(-kart_l / 2., -kart_w / 2., kart_l, kart_w)
    karts[str(index)].setPen(pg.mkPen(None))
    karts[str(index)].setBrush(pg.mkBrush(colors[index]))
    plots['2'].addItem(karts[str(index)])
    plots['2'].plot([0, 0], [0, 0], symbol='s', symbolBrush=colors[index], name=f'[{name[19:]}]')

    traces[str(index)] = plots['2'].plot(pen=colors[index])
    wheels_RL[str(index)] = plots['2'].plot(pen='k')
    wheels_RR[str(index)] = plots['2'].plot(pen='k')
    wheels_FL[str(index)] = plots['2'].plot(pen='k')
    wheels_FR[str(index)] = plots['2'].plot(pen='k')
    # turn_circle_vel[str(index)] = plots['2'].plot(pen='k')
    # turn_circle_steer[str(index)] = plots['2'].plot(pen='m')
    for i, topic in enumerate(plot_this):
        plots['1'].plot(data['time [s]'], data[topic], pen=colors[index], name=topic + f'[{name}]')
    color_count += 1

text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">This is</span><br><span style="color: #FF0; font-size: 16pt;">vx</span></div>', anchor=(1.4,0.5), angle=-45, border='w', fill=(0, 0, 255, 100))
plots['1'].addItem(text)
text.setPos(4, 7)
arrow = pg.ArrowItem(pos=(4, 7), angle=-135)
plots['1'].addItem(arrow)

text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">This is</span><br><span style="color: #FF0; font-size: 16pt;">vy</span></div>', anchor=(1.4,0.5), angle=45, border='w', fill=(0, 0, 255, 100))
plots['1'].addItem(text)
text.setPos(6.5, -0.2)
arrow = pg.ArrowItem(pos=(6.5, -0.2), angle=135)
plots['1'].addItem(arrow)

def updatePlot():
    # plots['2'].clear()
    for i in range(len(x_pos_data)):
        try:
            index = int(10 * plots['lr'].value())
            if index < 1:
                index = 1
            x = x_pos_data[i][:index]
            y = y_pos_data[i][:index]
            theta = theta_data[i][:index]
            beta = beta_data[i][index]
            vx = x_vel_data[i][index]
            vy = y_vel_data[i][index]
            transf = QtGui.QTransform()
            transf.translate(x[-1], y[-1])
            transf.rotate((theta[-1]) / np.pi * 180)
            transf.translate(0.135, 0)
            karts[str(i)].setTransform(transf)
            traces[str(i)].setData(x[-50:], y[-50:])
            posFL = [x[-1] + (kart_l / 2. + 0.135) * np.cos(theta[-1]) - kart_w / 2. * np.sin(theta[-1]),
                     y[-1] + kart_w / 2. * np.cos(theta[-1]) + (kart_l / 2. + 0.135) * np.sin(theta[-1])]
            posFR = [x[-1] + (kart_l / 2. + 0.135) * np.cos(theta[-1]) + kart_w / 2. * np.sin(theta[-1]),
                     y[-1] - kart_w / 2. * np.cos(theta[-1]) + (kart_l / 2. + 0.135) * np.sin(theta[-1])]
            wheels_RL[str(i)].setData(
                np.array([posFL[0] - (wheel_l + kart_l) * np.cos(theta[-1]),
                          posFL[0] - (-wheel_l + kart_l) * np.cos(theta[-1])]),
                np.array([posFL[1] - (wheel_l + kart_l) * np.sin(theta[-1]),
                          posFL[1] - (-wheel_l + kart_l) * np.sin(theta[-1])]))
            wheels_RR[str(i)].setData(
                np.array([posFR[0] - (wheel_l + kart_l) * np.cos(theta[-1]),
                          posFR[0] - (-wheel_l + kart_l) * np.cos(theta[-1])]),
                np.array([posFR[1] - (wheel_l + kart_l) * np.sin(theta[-1]),
                          posFR[1] - (-wheel_l + kart_l) * np.sin(theta[-1])]))
            wheels_FL[str(i)].setData(
                np.array([posFL[0] - wheel_l * np.cos(theta[-1] + beta), posFL[0] + wheel_l * np.cos(theta[-1] + beta)]),
                np.array([posFL[1] - wheel_l * np.sin(theta[-1] + beta), posFL[1] + wheel_l * np.sin(theta[-1] + beta)]))
            wheels_FR[str(i)].setData(
                np.array([posFR[0] - wheel_l * np.cos(theta[-1] + beta), posFR[0] + wheel_l * np.cos(theta[-1] + beta)]),
                np.array([posFR[1] - wheel_l * np.sin(theta[-1] + beta), posFR[1] + wheel_l * np.sin(theta[-1] + beta)]))

            # turn_circle_midpoint_vel = vx / vy * 0.46
            # turn_circle_midpoint_steer = 1.19 / np.tan(beta)
            # turn_circle_steer[str(i)].setData(
            #     np.array([x[-1] - 0.46 * np.cos(theta[-1]),
            #               x[-1] + turn_circle_midpoint_steer * np.cos(theta[-1] + np.pi / 2.0) - 0.46 * np.cos(theta[-1])]),
            #     np.array([y[-1] - 0.46 * np.sin(theta[-1]),
            #               y[-1] + turn_circle_midpoint_steer * np.sin(theta[-1] + np.pi / 2.0) - 0.46 * np.sin(theta[-1])]))
            # turn_circle_vel[str(i)].setData(
            #     np.array([x[-1] - 0.45 * np.cos(theta[-1]),
            #               x[-1] + turn_circle_midpoint_vel * np.cos(theta[-1] + np.pi / 2.0) - 0.45 * np.cos(theta[-1])]),
            #     np.array([y[-1] - 0.45 * np.sin(theta[-1]),
            #               y[-1] + turn_circle_midpoint_vel * np.sin(theta[-1] + np.pi / 2.0) - 0.45 * np.sin(theta[-1])]))
        except IndexError:
            pass

def play_pause():
    if button.isChecked():
        timer.start(dt*1000)
    else:
        timer.stop()


def moveline():
    if plots['lr'].value() < t_max:
        plots['lr'].setValue(plots['lr'].value() + dt)
    else:
        button.setChecked(False)
        timer.stop()
    # time.sleep(dt)


# def updateRegion():
#     plots['lr'].setRegion(plots['2'].getViewBox().viewRange()[0])

timer = pg.QtCore.QTimer()
timer.timeout.connect(moveline)
button.clicked.connect(play_pause)

plots['updatePlot'] = updatePlot
# plots['updateRegion'] = updateRegion
# plots['lr'].sigRegionChanged.connect(plots['updatePlot'])
plots['lr'].sigDragged.connect(plots['updatePlot'])
plots['lr'].sigPositionChanged.connect(plots['updatePlot'])
# plots['2'].sigXRangeChanged.connect(plots['updateRegion'])
plots['2'].showGrid(x=True, y=True, alpha=0.3)
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
