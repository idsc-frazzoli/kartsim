#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import numpy as np
import time
import threading

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt
import matplotlib.pyplot as plt

def main():

    real_time = False
    server_return_interval = 0.1  # [s] simulation time after which result is returned from server
    real_time_factor = 1
    _wait_for_real_time = 0

    if real_time:
        server_return_interval = 0.1*real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval*0.9*(1.0/real_time_factor)

    address = ('localhost', 6000)
    connected = False
    while not connected:
        try:
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            # print('ConnectionRefusedError')
            pass

    simTime = 10
    sim_time_increment = 0.01
    X0 = [0,
          0,
          0,
          0,
          0,
          0,
          0]

    u_time = np.linspace(0,10,1001)
    u_steering = np.linspace(-0.5,0.5,1001)

    u_brake = np.linspace(0,0,1001)
    u_mot_l = np.linspace(100,500,1001)
    u_mot_r = np.linspace(100,500,1001)

    U = np.array([u_time, u_steering, u_brake, u_mot_l, u_mot_r])

    U0 = U[:,:int(round(server_return_interval / sim_time_increment)) + 1]

    ticker = threading.Event()
    i = 0
    tgo = time.time()
    t0 = time.time()

    while not ticker.wait(_wait_for_real_time):
        # print('empty', time.time() - t0)
        # t0 = time.time()
        if i >= int(simTime / server_return_interval):
            conn.send('simulation finished')
            break
        elif i > 0:
            simRange = [int(round(i * server_return_interval / sim_time_increment)),
                        int(round((i + 1) * server_return_interval / sim_time_increment)) + 1]
            U0 = U[:,simRange[0]:simRange[1]]

        txt_msg = encode_request_msg_to_txt([X0, U0, server_return_interval, sim_time_increment])
        conn.send(txt_msg)

        answer_msg = conn.recv()

        X1 = decode_answer_msg_from_txt(answer_msg)
        X0 = list(X1[-1, :])
        i += 1
        # print('n', time.time() - t0)
        # t0 = time.time()

        # print(time.time()-t0)

    print('Success! Time overall: ', time.time()-tgo)

    # plt.plot(X1[:,0], X1[:,1])
    # plt.show()

if __name__ == '__main__':
    main()