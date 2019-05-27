#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 22.05.19 23:15

@author: mvb
"""
from multiprocessing.connection import Client
import threading
import numpy as np
import time
import pygame
import subprocess

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt

# first, run following command to connect racing wheel properly
# sudo usb_modeswitch -c /etc/usb_modeswitch.d/046d:c261

def main():
    pygame.init()
    pygame.joystick.init()

    try:
        logitech_wheel = pygame.joystick.Joystick(0)
    except pygame.error:
        print("Run following cmd to activate steering wheel: sudo usb_modeswitch -c /etc/usb_modeswitch.d/046d:c261")
        raise
    logitech_wheel.init()
    axes = logitech_wheel.get_numaxes()

    real_time = True
    server_return_interval = 1  # [s] simulation time after which result is returned from server
    real_time_factor = 1
    _wait_for_real_time = 0

    if real_time:
        server_return_interval = 0.05*real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval*0.9*(1.0/real_time_factor)
        print(server_return_interval,_wait_for_real_time)
    # preprodata = getpreprodata(pathpreprodata)

    connected = False
    while not connected:
        try:
            address = ('localhost', 6000)
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            print('ConnectionRefusedError')
            pass

    #___simulation parameters
    sim_time_increment = server_return_interval    # [s] time increment used in integration scheme inside simulation server (time step for logged simulation data)

    X0 = [0,0,0,0,0.2,0,0]

    # ______^^^______

    ticker = threading.Event()
    tgo = time.time()
    t0 = time.time()
    axes_init = False

    while not ticker.wait(_wait_for_real_time):

        for event in pygame.event.get():  # User did something
            pass
        logitech_wheel = pygame.joystick.Joystick(0)
        logitech_wheel.init()
        axex_values = [-logitech_wheel.get_axis(0)]
        for num in [1,3]:
            value = logitech_wheel.get_axis(num)
            if value > 0.98 or value == 0:
                value = 0
            else:
                value = -value + 1

            axex_values.append(value)
        # print('axes', axex_values)

        U = np.array([[-100, 100],
                      [axex_values[0], axex_values[0]],
                      [axex_values[1]-axex_values[2], axex_values[1]-axex_values[2]],
                      [axex_values[0], axex_values[0]],])

        # U = np.vstack((preprodata['time [s]'][simRange[0]:simRange[1]].values,
        #                  preprodata['MH BETA [rad]'][simRange[0]:simRange[1]].values,
        #                  preprodata['MH AB [m*s^-2]'][simRange[0]:simRange[1]].values,
        #                  preprodata['MH TV [rad*s^-2]'][simRange[0]:simRange[1]].values))
        #
        # # else:
        # #     tgo = time.time()
        #
        txt_msg = encode_request_msg_to_txt([X0, U, server_return_interval, sim_time_increment])
        # conn.send([X0, U, server_return_interval, sim_time_increment])

        conn.send(txt_msg)

        answer_msg = conn.recv()
        X1 = decode_answer_msg_from_txt(answer_msg)

        X0 = list(X1[-1,:])


    print('Success! Time overall: ', time.time()-tgo)

    print('connection closed')


if __name__ == '__main__':
    main()