#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:31:19 2019

@author: mvb
"""

from multiprocessing.connection import Client
from threading import Thread

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import time
import sys

#initializing Client and connect to pyKartsimServer
time.sleep(1)
try:
    port = int(sys.argv[1])
except:
    port = 6000
address = ('localhost', port+1)
connected = False
while not connected:
    try:
        conn = Client(address, authkey=b'kartSim2019')
        connected = True
    except ConnectionRefusedError:
        # print('ConnectionRefusedError')
        pass
##wait until initalization msg is sent and Simulation starts
#_ = conn.recv()
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(800,400)
win.setWindowTitle('kartsim visualization')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="xy", setAspectLocked = True)
#    p1.plot(x,y, pen=(255,255,255), name="Red curve")
p1.showGrid(x = True, y = True)
p1.setAspectLocked(lock=True, ratio=1)
#    
# p11 = win.addPlot(title="xy")
# p11.showGrid(x = True, y = True)
# p11.setAspectLocked(lock=True, ratio=1)

#
# p2 = win.addPlot(title="theta")
#
# win.nextRow()
#
# p3 = win.addPlot(title="vx")
#
# p4 = win.addPlot(title="vy")
#
# p5 = win.addPlot(title="vrot")
#
# win.nextRow()
#
# p6 = win.addPlot(title="slip angle")
#
# p7 = win.addPlot(title="AB")
#
# p8 = win.addPlot(title="TV")

kart_l = 1.19
kart_w = 1.0
wheel_l = 0.2

sim0 = 0
r1 = pg.QtGui.QGraphicsRectItem(-0.75, -0.5, 1.5, 1)
r1.setPen(pg.mkPen(None))
r1.setBrush(pg.mkBrush('r'))
#p1.setRange(xRange = [-2.5,2.5], yRange = [-2.5,2.5])
# p1.enableAutoRange('xy', False)
#p1.setRange(QtCore.QRect(23, 25, 37, 40))
X1 = np.array([[0,0,0,0,0,0,0,0,0,0]])
simTime = X1[:, 0]
x = X1[:, 1]
y = X1[:, 2]
theta = X1[:, 3]
vx = X1[:, 4]
vy = X1[:, 5]
vrot = X1[:, 6]
beta = X1[-1, 7]
beta_plot = X1[:, 7]
accRearAxle = X1[:, 8]
tv = X1[:, 9]
kart = pg.QtGui.QGraphicsRectItem(-kart_l / 2., -kart_w / 2., kart_l, kart_w)
kart.setPen(pg.mkPen(None))
kart.setBrush(pg.mkBrush('r'))
p1.addItem(kart)
# p1.plot([0, 0], [0, 0], pen=colors[-index - 1], name=f'[{name}]')

trace = p1.plot(pen='b')
wheel_RL = p1.plot(pen='k')
wheel_RR = p1.plot(pen='k')
wheel_FL = p1.plot(pen='k')
wheel_FR = p1.plot(pen='k')
axX1 = p1.getAxis('bottom')
axY1 = p1.getAxis('left')
# axX11 = p11.getAxis('bottom')
# axY11 = p11.getAxis('left')
# p1.setRange(QtCore.QRect(40, 30, 10, 25))

#     if axX11.range[0] < axX1.range[0] or axX11.range[1] > axX1.range[1] or axY11.range[0] < axY11.range[0] or axY11.range[1] > axY11.range[1]:
# #        p1.setRange(xRange = axX.range, yRange = axY.range)
#         p1.setRange(QtCore.QRect(axX11.range[0], axY11.range[0], axX11.range[1]-axX11.range[0], axY11.range[1]-axY11.range[0]))

# p11.plot(x, y, pen=(255, 255, 255))

# p2.clear()
# p2.plot(simTime,theta, pen=(255,255,255))
#
# p3.clear()
# p3.plot(simTime,vx, pen=(255,255,255))
#
# p4.clear()
# p4.plot(simTime,vy, pen=(255,255,255))
#
# p5.clear()
# p5.plot(simTime,vrot, pen=(255,255,255))

# p6.clear()
# p6.plot(simTime,np.arctan2(vy,vx), pen=(255,255,255))
# p6.plot(simTime,beta_plot, pen=(255,255,255))
#
# p7.clear()
# p7.plot(simTime,accRearAxle, pen=(255,255,255))
#
# p8.clear()
# p8.plot(simTime, tv, pen=(255, 255, 255))

runSimulation = True
#    first = True
X1 = []
def updateData():
    global X1
    try:
        conn.send('readyForNextMessage')
    except BrokenPipeError:
        timerdata.stop()
        raise BrokenPipeError
#    print('waiting for msg')
    msg = conn.recv()
    if len(msg) == 2:
        X = msg[0]
        U = msg[1]

        XU = np.concatenate((X, np.transpose(U)), axis=1)
    else:
        XU = msg

    if XU[0,0] == 'finished':
        pass
        # print('kill visualization')
        # print('timeOverall = ' + str(time.time() - t1))
        # timerdata.stop()

    elif XU[0,0] == 'init':
        t1 = time.time()
        X1 = []
    else:
        if len(X1) < 1:
            X1 = np.take(XU,[0,-1],axis=0)
        else:
            X1 = np.concatenate((X1,XU[-1:,:]))

        updatePlot(X1)

timerdata = QtCore.QTimer()
timerdata.timeout.connect(updateData)
timerdata.start()

            

def updatePlot(X1):
    simTime = X1[:,0]
    x = X1[:,1]
    y = X1[:,2]
    theta = X1[:,3]
    vx = X1[:,4]
    vy = X1[:,5]
    vrot = X1[:,6]
    beta = X1[-1,7]
    beta_plot = X1[:,7]
    accRearAxle = X1[:,8]
    tv = X1[:,9]
    transf = QtGui.QTransform()
    transf.translate(x[-1], y[-1])
    transf.rotate((theta[-1]) / np.pi * 180)
    transf.translate(0.135, 0)
    kart.setTransform(transf)
    trace.setData(x[-50:], y[-50:])
    posFL = [x[-1] + (kart_l / 2. + 0.135) * np.cos(theta[-1]) - kart_w / 2. * np.sin(theta[-1]),
             y[-1] + kart_w / 2. * np.cos(theta[-1]) + (kart_l / 2. + 0.135) * np.sin(theta[-1])]
    posFR = [x[-1] + (kart_l / 2. + 0.135) * np.cos(theta[-1]) + kart_w / 2. * np.sin(theta[-1]),
             y[-1] - kart_w / 2. * np.cos(theta[-1]) + (kart_l / 2. + 0.135) * np.sin(theta[-1])]
    wheel_RL.setData(
        np.array([posFL[0] - (wheel_l + kart_l) * np.cos(theta[-1]),
                  posFL[0] - (-wheel_l + kart_l) * np.cos(theta[-1])]),
        np.array([posFL[1] - (wheel_l + kart_l) * np.sin(theta[-1]),
                  posFL[1] - (-wheel_l + kart_l) * np.sin(theta[-1])]))
    wheel_RR.setData(
        np.array([posFR[0] - (wheel_l + kart_l) * np.cos(theta[-1]),
                  posFR[0] - (-wheel_l + kart_l) * np.cos(theta[-1])]),
        np.array([posFR[1] - (wheel_l + kart_l) * np.sin(theta[-1]),
                  posFR[1] - (-wheel_l + kart_l) * np.sin(theta[-1])]))
    wheel_FL.setData(
        np.array([posFL[0] - wheel_l * np.cos(theta[-1] + beta), posFL[0] + wheel_l * np.cos(theta[-1] + beta)]),
        np.array([posFL[1] - wheel_l * np.sin(theta[-1] + beta), posFL[1] + wheel_l * np.sin(theta[-1] + beta)]))
    wheel_FR.setData(
        np.array([posFR[0] - wheel_l * np.cos(theta[-1] + beta), posFR[0] + wheel_l * np.cos(theta[-1] + beta)]),
        np.array([posFR[1] - wheel_l * np.sin(theta[-1] + beta), posFR[1] + wheel_l * np.sin(theta[-1] + beta)]))


    # p11.clear()
    # p11.plot(x,y, pen=(255,255,255))
    
    # p2.clear()
    # p2.plot(simTime,theta, pen=(255,255,255))
    #
    # p3.clear()
    # p3.plot(simTime,vx, pen=(255,255,255))
    #
    # p4.clear()
    # p4.plot(simTime,vy, pen=(255,255,255))
    #
    # p5.clear()
    # p5.plot(simTime,vrot, pen=(255,255,255))

    # p6.clear()
    # p6.plot(simTime,np.arctan2(vy,vx), pen=(255,255,255))
    # p6.plot(simTime,beta_plot, pen=(255,255,255))
    #
    # p7.clear()
    # p7.plot(simTime,accRearAxle, pen=(255,255,255))
    #
    # p8.clear()
    # p8.plot(simTime, tv, pen=(255, 255, 255))
        
if __name__ == '__main__':
#    main()
    import sys
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        print('Plotting GUI doesn\'t exist \n')