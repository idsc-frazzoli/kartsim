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
from multiprocessing import Queue, Process
# from threading import Thread
import numpy as np
import sys
import time
from simulator.model.dynamic_mpc_model import DynamicVehicleMPC
from simulator.model.data_driven_model import DataDrivenVehicleModel
from simulator.model.hybrid_lstm_model import HybridLSTMModel
from simulator.integrate.systemequation import SystemEquation
from thread2 import Thread
import tensorflow as tf
sess = tf.Session()

def main():
    # global runThread, noThread, cliConn, logConn, vizConn
    # simulation default parameters
    try:
        port = int(sys.argv[1])
        visualization = int(sys.argv[2])
        logging = int(sys.argv[3])
        vehicle_model_type = sys.argv[4]
        vehicle_model_name = sys.argv[5]
    except:
        visualization = 0
        logging = 0
        vehicle_model_type = 'mpc_dynamic'
        vehicle_model_name = ''

    clientAddress = ('localhost', port)  # family is deduced to be 'AF_INET'
    clientListener = Listener(clientAddress, authkey=b'kartSim2019')
    if visualization:
        visualizationAddress = ('localhost', port+1)  # family is deduced to be 'AF_INET'
        visualizationListener = Listener(visualizationAddress, authkey=b'kartSim2019')
    logAddress = ('localhost', port+2)  # family is deduced to be 'AF_INET'
    logListener = Listener(logAddress, authkey=b'kartSim2019')

    noThread = True
    vizConn = None
    logConn = None
    cliConn = None
    runServer = True

    # initialize vehicle model
    # vehicle_model = AccelerationReferenceModel()
    if vehicle_model_type == 'mpc_dynamic':
        vehicle_model = DynamicVehicleMPC()
    elif vehicle_model_type == 'hybrid_mlp':
        vehicle_model = DataDrivenVehicleModel(model_name=vehicle_model_name)
    elif vehicle_model_type == 'hybrid_lstm':
        vehicle_model = HybridLSTMModel(model_name=vehicle_model_name)

    # vehicle_model = DynamicVehicleMPC(direct_input=True)

    # system_equation = SystemEquation(vehicle_model)

    while runServer:
        if vehicle_model_type == 'hybrid_lstm':
            vehicle_model.reinitialize_variables()
        system_equation = SystemEquation(vehicle_model)
        if visualization and vizConn is None:
            # print("waiting for visualization connection at", visualizationAddress)
            vizConn = visualizationListener.accept()
            print('visualization connection accepted from', visualizationListener.last_accepted)
        else:
            pass

        if logging and logConn is None:
            # print("waiting for logger connection at", logAddress)
            logConn = logListener.accept()
            print('logger connection accepted from', logListener.last_accepted)
        else:
            pass
        if cliConn is None:
            # print("waiting for client connection at", clientAddress)
            cliConn = clientListener.accept()
            print('client connection accepted from', clientListener.last_accepted)
            print('Starting simulation:\n')

        cliConn, vizConn, logConn = interprocess_communication(cliConn, vizConn, logConn,
                                                                         visualization, logging, system_equation, )

def interprocess_communication(cliConn, vizConn, logConn, visualization, logging, system_equation, ):
    initSignal = 0
    runThread = True
    result_queue = Queue()
    while runThread:
        try:
            request_msg = cliConn.recv()
            msg_list = request_msg.split("\n")
            if len(msg_list) == 4:
                X0, U, server_return_interval, sim_time_increment = decode_request_msg_from_txt(request_msg)

                numerical_integration = Thread(target=execute_integration,
                                                args=(X0, U, server_return_interval, sim_time_increment,
                                                      system_equation, result_queue))
                numerical_integration.start()

                patience = server_return_interval * 10
                if patience < 10:
                    patience = 10
                numerical_integration.join(patience)
                if numerical_integration.is_alive():
                    if visualization:
                        vizConn.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                    if logging:
                        logConn.send(np.array([['abort', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                    answer_msg = encode_answer_msg_to_txt(
                        'Numerical integration unstable and timed out. Abort simulation.')
                    cliConn.send(answer_msg)
                    try:
                        numerical_integration.terminate()
                    except RuntimeError:
                        pass
                    return cliConn, vizConn, logConn
                X = result_queue.get()
                # print('server send', X[-1])
                answer_msg = encode_answer_msg_to_txt(X)
                # print('server send', answer_msg)
                cliConn.send(answer_msg)
                if 'failed' in answer_msg:
                    if visualization:
                        vizConn.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                    if logging:
                        logConn.send(np.array([['abort', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                    return cliConn, vizConn, logConn

            elif request_msg == 'simulation finished':
                if visualization:
                    vizConn.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                if logging:
                    logConn.send(np.array([['finished', 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                return cliConn, vizConn, logConn
            else:
                print('FormatError: wrong message format.')
                raise ValueError
        except EOFError:
            # print('SimClientError: BrokenPipe', cliConn.fileno())
            cliConn = None
            return cliConn, vizConn, logConn

        if logging:
            try:
                logConn.send([X, U[1:]])
            except:
                print('LoggerError: BrokenPipe', logConn.fileno())
                logConn = None
                return cliConn, vizConn, logConn
        #                print('sendTime logConn: ', time.time() - tt)
        if visualization:
            try:
                if initSignal < 1:
                    vizConn.send(np.array([['init', 0, 0, 0, 1, 0, 0, 0.5, 0, 0]]))
                    initSignal = 1
                if vizConn.poll():
                    vizConn.recv()
                    vizConn.send([X, U[1:]])
            except:
                print('VisualizationError: BrokenPipe', vizConn.fileno())
                vizConn = None
                return cliConn, vizConn, logConn


def execute_integration(X0, U, server_return_interval, sim_time_increment, system_equation, result_queue):
    X = integrators.odeIntegratorIVP(X0, U, server_return_interval, sim_time_increment, system_equation)
    # X = integrators.euler(X0, U, server_return_interval, sim_time_increment)
    result_queue.put(X)


if __name__ == '__main__':
    # tf.keras.backend.set_session(sess)
    main()
