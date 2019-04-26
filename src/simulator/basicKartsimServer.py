#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

import simulator.timeIntegrators as integrator
from multiprocessing.connection import Listener
from threading import Thread, active_count
import numpy as np
import sys
import time

def main():
    global runThread

    #simulation default parameters
    simIncrement = 0.01  #s underlying time step for integration

    try:
        visualization = int(sys.argv[1])
        logging = int(sys.argv[2])
    except:
        visualization = 0
        logging = 1


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

            time.sleep(1)
#            print('No of active threads running: ', active_count())
            print("waiting for client connection at", clientAddress)
            cliConn = clientListener.accept()
            print('client connection accepted from', clientListener.last_accepted)
            
            t = Thread(target=handle_client, args=(cliConn,vizConn,logConn,simIncrement,visualization,logging,))
            runThread = True
            t.start()
        else:
            time.sleep(0.1)
            pass
            # if active_count() < 2:
            #     noThread = True


def handle_client(c,v,l,simIncrement,visualization,logging):
    initSignal = 0
    while runThread:
        try:

            msg = c.recv()

            if len(msg) == 3 and len(msg[0]) == 7 and isinstance(msg[2], float):
                X0 = msg[0]
                U = msg[1]
                simStep = msg[2]
                # X = integrator.odeIntegrator(X0, U, simStep, simIncrement) #format: X0 = [simTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]; X0 = [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0]
                X = integrator.odeIntegratorIVP(X0, U, simStep, simIncrement) #format: X0 = [simTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]; X0 = [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0]

                c.send(X)

            else:
                print('FormatError: msg sent to server must be of form:\n   msg = [[simStartTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv],[simTimeStep]]\n e.g. msg = [[0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0],[0.1]]')
        except EOFError:
            # print('EOFError: exit thread', c.fileno())
            if visualization:
                v.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                initSignal = 0
            if logging:
                l.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
            break


        if logging:
            try:
                l.send([X,U[1:]])
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
                    v.send([X,U[1:]])
            except:
                print('VisualizationError: BrokenPipe', v.fileno())
                break
    
if __name__ == '__main__':
    main()