#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

import matlab.engine
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import time


#p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")

#def main():
eng = None
print('Looking for shared Matlab session...\n')
name = matlab.engine.find_matlab()
print('name = ' + str(name) + '\n')
if len(name) > 0:
    print('Shared Matlab session ' + str(name[0]) + ' found.')
    try:
        eng = matlab.engine.connect_matlab(name[0])
        print('Connected to session ' + str(name[0]) + '.')
    except:
        print('Could not connect to session ' + str(name[0]) + '.')
        eng = None
else:
    print('No shared Matlab session found.\nCreating new normal session...')
    eng = matlab.engine.start_matlab()
#    try:
#        eng
#    except NameError: 
#        eng = None
#        print('No Matlab session connected!')
#    eng = None
#    if eng == None:
#        print('Connecting to shared Matlab session...')
#        eng = matlab.engine.connect_matlab(name=None)#, background=True)
##        eng = matlab.engine.connect_matlab('MATLAB_6409')
#eng = matlab.engine.start_matlab()
if eng == None:
    print('No Matlab session exists! Please start a Matlab session \n')
else:
    print('Calculations start...')
    
    B = 4
    C = 1.7
    D = 0.7*9.81
    Cf = 0.15
    B1 = B
    B2 = B
    C1 = C
    C2 = C
    D1 = 0.8*D
    D2 = D
    maxA = D*0.9
    param = [B1,C1,D1,B2,C2,D2,Cf,maxA]
    
    #initial values
    x = [0]
    y = [0]
    theta = [0]
    vx = [1]
    vy = [0]
    vrot = [0]
    beta = 0.5          #steering angles
    accRearAxle = 2     #acceleration at rear axle
    tv = -0.5              #specific force difference at rear tires
    
    #simulation
    timeStep = 0.005    #s
    simTime = 20        #s
    

        

    app = QtGui.QApplication([])

    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph example: Plotting')
    
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    p1 = win.addPlot(title="xy", setAspectLocked = True)
#    p1.plot(x,y, pen=(255,255,255), name="Red curve")
    p1.showGrid(x = True, y = True)
    p1.setAspectLocked(lock=True, ratio=1)
        
    p11 = win.addPlot(title="xy")
    p11.showGrid(x = True, y = True)
    p11.setAspectLocked(lock=True, ratio=1)

        
    p2 = win.addPlot(title="theta")
#    
#    
#    win.nextRow()
#    
#    p3 = win.addPlot(title="vx")
#    p3.plot(vx, pen=(255,255,255), name="Red curve")
#    
#    p4 = win.addPlot(title="vy")
#    p4.plot(vy, pen=(255,255,255), name="Red curve")
#    
#    p5 = win.addPlot(title="vrot")
#    p5.plot(vrot, pen=(255,255,255), name="Red curve")

    

#    ptr = 0
#    t0 = time.time()
#    axX = p1.getAxis('bottom')
#    axY = p1.getAxis('left')
#    rangeX = axX.range
#    rangeY = axY.range
#    xScale = 1.
#    yScale = 1.
#    rangeXtot = rangeX[1]-rangeX[0]
#    rangeYtot = rangeY[1]-rangeY[0]
#    if rangeXtot < rangeYtot:
#        xScale = rangeYtot/rangeXtot
#    else:
#        yScale = rangeXtot/rangeYtot
##    p1.setXRange = (rangeX[0]*xScale,rangeX[1]*xScale)
##    p1.setXRange = (rangeY[0]*yScale,rangeY[1]*yScale)
##    p1.enableAutoRange('xy', False)
    
    #    for i in range(int(simTime/timeStep)):
    kart_l = 1.5
    kart_w = 1. 
    
    t1 = time.time()
    t0 = time.time()
    i = 0
    transf = QtGui.QTransform()
    r1 = pg.QtGui.QGraphicsRectItem(-0.75, -0.5, 1.5, 1)
    r1.setPen(pg.mkPen(None))
    r1.setBrush(pg.mkBrush('r'))
    p1.setRange(xRange = [-2.5,2.5], yRange = [-2.5,2.5])
    p1.enableAutoRange('xy', False)   
    print('hello')
    def update():
        global i, t0, t1
        print('timeOverall = ' + str(time.time()-t1))
        t1 = time.time()
#        try:
        print('time1 = ' + str(time.time()-t0))
        t0 = time.time()
        t = eng.modelDx_pymod(vx[-1],vy[-1],vrot[-1],beta,accRearAxle,tv, param, nargout=3)
        print('time2 = ' + str(time.time()-t0))
        t0 = time.time()
        x_new, y_new, theta_new, vx_new, vy_new, vrot_new = integrator (x[-1], y[-1], theta[-1], vx[-1], vy[-1], vrot[-1], t[0], t[1], t[2], timeStep)
        print('time3 = ' + str(time.time()-t0))
        t0 = time.time()
        vx.append(vx_new)
        vy.append(vy_new)
        vrot.append(vrot_new)
        x.append(x_new)
        y.append(y_new)
        theta.append(theta_new)
        print('time4 = ' + str(time.time()-t0))
        t0 = time.time()
        updatePlot()
        print('time4.1 = ' + str(time.time()-t0))
        t0 = time.time()
        i += 1
            
#        finally:
#            timer.singleShot(0,update)
#            print('time5 = ' + str(time.time()-t0))
#            t0 = time.time()
#        except:
#            print('Something went wrong!')
#            timer.stop()
        
#        print('accel  ',t[0],t[1],t[2])
            

    print('Done')

            
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(timeStep*1000))
    
def updatePlot():
    p1.clear()
    r1 = pg.QtGui.QGraphicsRectItem(-kart_l/2., -kart_w/2., kart_l, kart_w)
    r1.setPen(pg.mkPen(None))
    r1.setBrush(pg.mkBrush('r'))
#        transf.rotate((vrot[-1]* timeStep)/np.pi*180+180)
    transf.translate(vx[-1]*timeStep, vy[-1]*timeStep)
    transf.rotate((vrot[-1]* timeStep)/np.pi*180)
    r1.setTransform(transf)
    p1.addItem(r1)
    axX = p11.getAxis('bottom')
    axY = p11.getAxis('left')
    if axX.range[0] < -2 or axX.range[1] > 2 or axY.range[0] < -2 or axY.range[1] > 2:
        p1.setRange(xRange = axX.range + [-2,2], yRange = axY.range + [-2,2])
#        arrow2 = pg.ArrowItem(pos=(x[-1],y[-1]),angle=(-theta[-1]/np.pi*180+180))
#        p1.addItem(arrow2)
    posFL = [x[-1]+kart_l/2.*np.cos(theta[-1])-kart_w/2.*np.sin(theta[-1]), y[-1]+kart_w/2.*np.cos(theta[-1])+kart_l/2.*np.sin(theta[-1])]
    posFR = [x[-1]+kart_l/2.*np.cos(theta[-1])+kart_w/2.*np.sin(theta[-1]), y[-1]-kart_w/2.*np.cos(theta[-1])+kart_l/2.*np.sin(theta[-1])]
    wheel_l = 0.2
    p1.plot([posFL[0] - wheel_l * np.cos(theta[-1] + beta), posFL[0] + wheel_l * np.cos(theta[-1] + beta)],[posFL[1] - wheel_l * np.sin(theta[-1] + beta), posFL[1] + wheel_l * np.sin(theta[-1] + beta)])
    p1.plot([posFR[0] - wheel_l * np.cos(theta[-1] + beta), posFR[0] + wheel_l * np.cos(theta[-1] + beta)],[posFR[1] - wheel_l * np.sin(theta[-1] + beta), posFR[1] + wheel_l * np.sin(theta[-1] + beta)])
    
    p11.clear()
    p11.plot(x[:-1],y[:-1], pen=(255,0,0), name="Red curve")
    
#    p2.clear()
#    p2.plot(theta[:-1], pen=(255,255,255), name="Red curve")
    
def integrator (x, y, theta, vx, vy, vrot, accX, accY, accRot, timeStep):
    vx = vx + accX * timeStep
    vy = vy + accY * timeStep
    vrot = vrot + accRot * timeStep
    x = x + (vx * np.cos(theta) - vy * np.sin(theta)) * timeStep
    y = y + (vy * np.cos(theta) + vx * np.sin(theta)) * timeStep
    theta = theta + vrot * timeStep
    return x,y,theta,vx,vy,vrot

    
if __name__ == '__main__':
    import sys
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        print('Plotting GUI doesn\'t exist \n')
                
    
