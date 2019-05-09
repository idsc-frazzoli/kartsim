#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

from integrate import timeIntegrators as integrator
from multiprocessing.connection import Listener
from threading import Thread, active_count
import numpy as np


def main():
    global runThread
    
    #simulation default parameters
    simIncrement = 0.01  #s underlying time step for integration
    visualization = True
    logging = True
    clientAddress = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    clientListener = Listener(clientAddress, authkey=b'kartSim2019')
    if visualization:
        visualizationAddress = ('localhost', 6001)     # family is deduced to be 'AF_INET'
        visualizationListener = Listener(visualizationAddress, authkey=b'kartSim2019')
    logAddress = ('localhost', 6002)     # family is deduced to be 'AF_INET'
    logListener = Listener(logAddress, authkey=b'kartSim2019')
    
    noThread = True
    vizConn = None
    logConn = None
    while True:
        if noThread:
            noThread = False
            
            if visualization and vizConn is None:
                print("waiting for visualization connection at", visualizationAddress)
                vizConn = visualizationListener.accept()
                print('visualization connection accepted from', visualizationListener.last_accepted)
            else:
                pass
            
            if logging and logConn is None:
                print("waiting for logger connection at", logAddress)
                logConn = logListener.accept()
                print('logger connection accepted from', logListener.last_accepted)
            else:
                pass
            
#            print('No of active threads running: ', active_count())
            print("waiting for client connection at", clientAddress)
            cliConn = clientListener.accept()
            print('client connection accepted from', clientListener.last_accepted)
            
            t = Thread(target=handle_client, args=(cliConn,vizConn,logConn,simIncrement,visualization,logging,))
            runThread = True
            t.start()
        else:
            if active_count() < 2:
                noThread = True
    

def handle_client(c,v,l,simIncrement,visualization,logging):
    initSignal = 0
    while runThread:
        try:
            msg = c.recv()
            if len(msg) == 2 and len(msg[0]) == 10 and isinstance(msg[1], float):
                X0 = msg[0]
                simStep = msg[1]
                
                X = integrator.odeIntegrator(X0, simStep, simIncrement) #format: X0 = [simTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]; X0 = [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0]
#                tt = time.time()
                c.send(X)
#                print('sendTime c: ', time.time() - tt)
#                tt = time.time()
            else:
                print('FormatError: msg sent to server must be of form:\n   msg = [[simStartTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv],[simTimeStep]]\n e.g. msg = [[0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0],[0.1]]')
        except EOFError:
            print('EOFError: exit thread', c.fileno())
            if visualization:
                v.send(np.array([['finished', 0, 0, 0, 1, 0, 0, 0.5, 0, 0]]))
                initSignal = 0
            if logging:
                l.send(np.array([['finished', 0, 0, 0, 1, 0, 0, 0.5, 0, 0]]))
            break
        
        if logging:
            try:
                l.send(X)
            except:
                print('LoggerError: BrokenPipe', l.fileno())
                break
#                print('sendTime l: ', time.time() - tt)

        if visualization:
            try:
                if initSignal < 1:
                    v.send(np.array([['init', 0, 0, 0, 1, 0, 0, 0.5, 0, 0]]))
                    initSignal = 1
                if v.poll():
                    v.recv()
                    v.send(X)
            except:
                print('VisualizationError: BrokenPipe', v.fileno())
                break
        

    
if __name__ == '__main__':
    main()