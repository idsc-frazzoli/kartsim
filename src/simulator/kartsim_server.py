#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:49 2019

@author: mvb
"""

# import simulator.timeIntegrators as integrators
import integrate.timeIntegrators as integrators
from textcommunication import decode_request_msg_from_txt, encode_answer_msg_to_txt
from multiprocessing.connection import Listener
from threading import Thread
import numpy as np
import sys
import time

def main():
    global runThread, noThread, cliConn, logConn, vizConn

    #simulation default parameters

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
    cliConn = None
    runServer = True

    while runServer:
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
            if cliConn is None:
                print("waiting for client connection at", clientAddress)
                cliConn = clientListener.accept()
                print('client connection accepted from', clientListener.last_accepted)
                print('Starting simulation:\n')
            
            t = Thread(target=handle_client, args=(cliConn,vizConn,logConn,visualization,logging,))
            runThread = True
            t.start()
        else:
            time.sleep(0.1)
            # if active_count() < 2:
            #     noThread = True
            pass


def handle_client(c,v,l,visualization,logging):
    global noThread, cliConn, logConn, vizConn
    initSignal = 0
    while runThread:
        try:
            request_msg = c.recv()
            msg_list = request_msg.split("\n")
            if len(msg_list) == 4:
                X0, U, server_return_interval, sim_time_increment = decode_request_msg_from_txt(request_msg)

                # X = integrators.odeIntegrator(X0, U, server_return_time, sim_time_increment) #format: X0 = [simTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]; X0 = [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0]
                X = integrators.odeIntegratorIVP(X0, U, server_return_interval, sim_time_increment) #format: X0 = [simTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]; X0 = [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0]

                answer_msg = encode_answer_msg_to_txt(X)
                # time.sleep(0.01)
                c.send(answer_msg)

            elif request_msg == 'simulation finished':
                noThread = True
                if visualization:
                    v.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                if logging:
                    l.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                break
            else:
                print('FormatError: msg sent to server must be of form:\n   msg = [[simStartTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv],[simTimeStep]]\n e.g. msg = [[0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0],[0.1]]')
        except EOFError:
            print('SimClientError: BrokenPipe', c.fileno())
            cliConn = None
            noThread = True
            break


        if logging:
            try:
                l.send([X,U[1:]])
            except:
                print('LoggerError: BrokenPipe', l.fileno())
                logConn = None
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
                vizConn = None
                break
    
if __name__ == '__main__':
    main()