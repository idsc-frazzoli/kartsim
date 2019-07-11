    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import time
from integrate import timeIntegrators as integrator

#p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")

#def main():
    
#initial state
t0 = 0
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
win.setWindowTitle('showsimdata example: Plotting')

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
X1 = np.zeros((1,10))
rightNow = [0]
def update():
    global i, t1, X0, X1, simTime, rightNow, sim0
    print('i = ' + str(i))
    try:
        if i == 0:
            X0 = [t0, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
    
        X = integrator.odeIntegrator(X0, vizStep, simStep)
        
        X0 = list(X[-1,:])
        X1 = np.concatenate((X1,X[1:,:]))
#        x_new, y_new, theta_new, vx_new, vy_new, vrot_new = integrator.euler (x[-1], y[-1], theta[-1], vx[-1], vy[-1], vrot[-1], t[0], t[1], t[2], simStep)
    
        
        
        print('timeOverall = ' + str(time.time() - t1))
        t1 = time.time()
        print('simTime' + str(X[-1,0]-sim0))
        sim0 = X[-1,0]
#        ts = np.linspace(0,vizStep,int(vizStep/simStep)+1)
#        simTime = np.concatenate((simTime,np.array([ts[1:]+rightNow]).transpose()))
#        rightNow = simTime[-1][0]
        
        
        updatePlot(X1)
    
        i += 1
        
#    finally:
#        timer.singleShot(0,update)

    except:
        print('Something went wrong!')
        timer.stop()
        

print('Done')

X0 = [t0, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]

#simulation
simStep = 0.05    #s
vizStep = 0.05        #s
simTime = np.array([[0]])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(vizStep*1000))
    
def updatePlot(X1):
    simTime = X1[:,0]
    x = X1[:,1]
    y = X1[:,2]
    theta = X1[:,3]
    vx = X1[:,4]
    vy = X1[:,5]
    vrot = X1[:,6]
    
    p1.clear()
    r1 = pg.QtGui.QGraphicsRectItem(-kart_l/2., -kart_w/2., kart_l, kart_w)
    r1.setPen(pg.mkPen(None))
    r1.setBrush(pg.mkBrush('r'))
    transf = QtGui.QTransform()
    transf.translate(x[-1], y[-1])
    transf.rotate((theta[-1])/np.pi*180)
    r1.setTransform(transf)
    p1.addItem(r1)
    axX = p11.getAxis('bottom')
    axY = p11.getAxis('left')
    if axX.range[0] < -2 or axX.range[1] > 2 or axY.range[0] < -2 or axY.range[1] > 2:
        p1.setRange(xRange = axX.range + [-2,2], yRange = axY.range + [-2,2])
    posFL = [x[-1]+kart_l/2.*np.cos(theta[-1])-kart_w/2.*np.sin(theta[-1]), y[-1]+kart_w/2.*np.cos(theta[-1])+kart_l/2.*np.sin(theta[-1])]
    posFR = [x[-1]+kart_l/2.*np.cos(theta[-1])+kart_w/2.*np.sin(theta[-1]), y[-1]-kart_w/2.*np.cos(theta[-1])+kart_l/2.*np.sin(theta[-1])]
    wheel_l = 0.2
    p1.plot([posFL[0] - wheel_l * np.cos(theta[-1] + beta), posFL[0] + wheel_l * np.cos(theta[-1] + beta)],[posFL[1] - wheel_l * np.sin(theta[-1] + beta), posFL[1] + wheel_l * np.sin(theta[-1] + beta)])
    p1.plot([posFR[0] - wheel_l * np.cos(theta[-1] + beta), posFR[0] + wheel_l * np.cos(theta[-1] + beta)],[posFR[1] - wheel_l * np.sin(theta[-1] + beta), posFR[1] + wheel_l * np.sin(theta[-1] + beta)])
    
    p11.clear()
    p11.plot(x,y, pen=(255,255,255))
    
    p2.clear()
    p2.plot(simTime,theta, pen=(255,255,255))
    
    p3.clear()
    p3.plot(simTime,vx, pen=(255,255,255))
    
    p4.clear()
    p4.plot(simTime,vy, pen=(255,255,255))
    
    p5.clear()
    p5.plot(simTime,vrot, pen=(255,255,255))
    
    p6.clear()
    p6.plot(simTime,np.arctan2(vy,vx), pen=(255,255,255))

    
if __name__ == '__main__':
    import sys
    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    except:
        print('Plotting GUI doesn\'t exist \n')
                
    
