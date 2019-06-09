#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import numpy as np
import time

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt
import matplotlib.pyplot as plt

def main():

    connected = False
    while not connected:
        try:
            address = ('localhost', 6000)
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            print('ConnectionRefusedError')
            pass

    simTime = 10
    sim_time_increment = 0.1
    X0 = [0,
          0,
          0,
          0,
          1,
          0,
          0]

    u_time = np.linspace(0,10,100)
    u_BETA = np.linspace(-0.5,0.5,100)

    AB_const = np.linspace(1,1,100)
    TV_const = np.linspace(0,0,100)

    U = np.array([u_time, u_BETA, AB_const, TV_const])

    tgo = time.time()
    txt_msg = encode_request_msg_to_txt([X0, U, simTime, sim_time_increment])
    # conn.send([X0, U, server_return_interval, sim_time_increment])
    conn.send(txt_msg)

    answer_msg = conn.recv()
    X1 = decode_answer_msg_from_txt(answer_msg)

    print(X1)

    print('Success! Time overall: ', time.time()-tgo)

    plt.plot(X1[:,1], X1[:,2])
    plt.show()

if __name__ == '__main__':
    main()