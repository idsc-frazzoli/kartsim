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
from collections import deque

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
        logitech_wheel = None
        # raise
    if logitech_wheel != None:
        logitech_wheel.init()
        axes = logitech_wheel.get_numaxes()

    real_time = True
    server_return_interval = 1  # [s] simulation time after which result is returned from server
    real_time_factor = 1
    _wait_for_real_time = 0

    if real_time:
        time_step = 0.05
        server_return_interval = time_step * real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval * 0.9 * (1.0 / real_time_factor)
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

    # ___simulation parameters
    sim_time_increment = server_return_interval  # [s] time increment used in integration scheme inside simulation server (time step for logged simulation data)

    X0 = [0, 0, 0, 0, 0, 0, 0]

    # ______^^^______

    ticker = threading.Event()

    time_steps = deque([time.time() - time_step, time.time()], maxlen=2)
    wheel_axis = deque([0, 0], maxlen=2)
    pedals = [0, 0]

    while not ticker.wait(_wait_for_real_time):
        for event in pygame.event.get():  # User did something
            # print(event)
            pass
        # event = pygame.event.wait()
        # logitech_wheel = pygame.joystick.Joystick(0)
        # logitech_wheel.init()
        wheel_axis.append(-logitech_wheel.get_axis(0))
        time_steps.append(time.time())
        for i, num in enumerate([1, 3]):
            value = logitech_wheel.get_axis(num)
            if value > 0.98 or value == 0:
                value = 0
            else:
                value = -value + 1

            pedals[i] = value
        # print(np.array(wheel_axis)-100.0, end='\r')
        # print(np.array(time_steps)-time_steps[0], end='\r')
        pressed = pygame.key.get_pressed()
        # print(pressed)
        if pressed[pygame.K_w]:
            print("w is pressed")
        if pressed[pygame.K_s]:
            print("s is pressed")
        # print('axes', axex_values)
        # print(list(wheel_axis)[-1])
        U = np.array([list(np.array(time_steps) - time_steps[0] + X0[0]),
                      list(np.array(wheel_axis)*3),
                      [pedals[0] - pedals[1]*4, pedals[0] - pedals[1]*4],
                      list(wheel_axis), ])

        txt_msg = encode_request_msg_to_txt([X0, U, server_return_interval, sim_time_increment])
        # conn.send([X0, U, server_return_interval, sim_time_increment])

        conn.send(txt_msg)

        answer_msg = conn.recv()
        X1 = decode_answer_msg_from_txt(answer_msg)

        X0 = list(X1[-1, :])

    print('connection closed')


if __name__ == '__main__':
    main()
