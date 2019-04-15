#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

import matlab.engine
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import time
import timeIntegrators as integrator


#p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")

#def main():
#___________________Connect to Matlab Session________________________
#eng = None
#print('Looking for shared Matlab session...\n')
#name = matlab.engine.find_matlab()
#print('name = ' + str(name) + '\n')
#if len(name) > 0:
#    print('Shared Matlab session ' + str(name[0]) + ' found.')
#    try:
#        eng = matlab.engine.connect_matlab(name[0])
#        print('Connected to session ' + str(name[0]) + '.')
#    except:
#        print('Could not connect to session ' + str(name[0]) + '.')
#        eng = None
#else:
#    print('No shared Matlab session found.\nCreating new normal session...')
#    eng = matlab.engine.start_matlab()
#
#
#if eng == None:
#    print('No Matlab session exists! Please start a Matlab session \n')
#else:
#    print('Calculations start...')
#^^^_______________Connect to Matlab Session____________________^^^
    
    
#    B = 4
#    C = 1.7
#    D = 0.7*9.81
#    Cf = 0.15
#    B1 = B
#    B2 = B
#    C1 = C
#    C2 = C
#    D1 = 0.8*D
#    D2 = D
#    maxA = D*0.9
#    param = [B1,C1,D1,B2,C2,D2,Cf,maxA]
    
#initial values
x = 0
y = 0
theta = 0
vx = 0
vy = 0
vrot = 0
beta = 0.5              #steering angles
accRearAxle = 1.8       #acceleration at rear axle
tv = 0.4                #specific force difference at rear tires


app = QtGui.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1600,800)
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

win.nextRow()

p3 = win.addPlot(title="vx")

p4 = win.addPlot(title="vy")

p5 = win.addPlot(title="vrot")

win.nextRow()

p6 = win.addPlot(title="slip angle")

kart_l = 1.5
kart_w = 1. 

t1 = time.time()
sim0 = 0
r1 = pg.QtGui.QGraphicsRectItem(-0.75, -0.5, 1.5, 1)
r1.setPen(pg.mkPen(None))
r1.setBrush(pg.mkBrush('r'))
p1.setRange(xRange = [-2.5,2.5], yRange = [-2.5,2.5])
p1.enableAutoRange('xy', False)       
i = 0
X1 = np.zeros((1,9))
rightNow = [0]
def update():
    global i, t1, X0, X1, simTime, rightNow, sim0
    print('i = ' + str(i))
    try:
        if i == 0:
            X0 = [x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
    
        X = integrator.odeIntegrator(X0, vizStep, simStep)
        
#        print(X1)
        X0 = list(X[-1,:])
        X1 = np.concatenate((X1,X[1:,:]))
#        x_new, y_new, theta_new, vx_new, vy_new, vrot_new = integrator.euler (x[-1], y[-1], theta[-1], vx[-1], vy[-1], vrot[-1], t[0], t[1], t[2], simStep)
    
        
        
        print('timeOverall = ' + str(time.time() - t1))
#        t1 = time.time()
        print('simTime' + str(simTime[-1,0]-sim0))
#        sim0 = simTime[-1,0]
        ts = np.linspace(0,vizStep,int(vizStep/simStep)+1)
        simTime = np.concatenate((simTime,np.array([ts[1:]+rightNow]).transpose()))
        rightNow = simTime[-1][0]
        
        
        updatePlot(X1,X)
    
        i += 1
        
#    finally:
#        timer.singleShot(0,update)

    except:
        print('Something went wrong!')
        timer.stop()
        

print('Done')

X0 = [x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]

#simulation
simStep = 0.05    #s
vizStep = 0.05        #s
simTime = np.array([[0]])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(vizStep*1000))
    
def updatePlot(X1,X):
    x = X1[:,0]
    y = X1[:,1]
    theta = X1[:,2]
    vx = X1[:,3]
    vy = X1[:,4]
    vrot = X1[:,5]
    
    p1.clear()
    r1 = pg.QtGui.QGraphicsRectItem(-kart_l/2., -kart_w/2., kart_l, kart_w)
    r1.setPen(pg.mkPen(None))
    r1.setBrush(pg.mkBrush('r'))
#        transf.rotate((vrot[-1]* simStep)/np.pi*180+180)

    transf = QtGui.QTransform()
    transf.translate(x[-1], y[-1])
    transf.rotate((theta[-1])/np.pi*180)
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
    p11.plot(x,y, pen=(255,255,255))
    
    p2.clear()
    p2.plot(simTime[:,0],theta, pen=(255,255,255))
    
    p3.clear()
    p3.plot(simTime[:,0],vx, pen=(255,255,255))
    
    p4.clear()
    p4.plot(simTime[:,0],vy, pen=(255,255,255))
    
    p5.clear()
    p5.plot(simTime[:,0],vrot, pen=(255,255,255))
    
    p6.clear()
    p6.plot(simTime[:,0],np.arctan2(vy,vx), pen=(255,255,255))

    
#def integrator (x, y, theta, vx, vy, vrot, accX, accY, accRot, simStep):
#    vx = vx + accX * simStep
#    vy = vy + accY * simStep
#    vrot = vrot + accRot * simStep
#    x = x + (vx * np.cos(theta) - vy * np.sin(theta)) * simStep
#    y = y + (vy * np.cos(theta) + vx * np.sin(theta)) * simStep
#    theta = theta + vrot * simStep
#    return x,y,theta,vx,vy,vrot

    
if __name__ == '__main__':
    import sys
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        print('Plotting GUI doesn\'t exist \n')
                
    
