#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 23:17

@author: mvb
"""
import numpy as np


def decode_request_msg_from_txt(msg):
    msg_list = msg.split("\n")
    if len(msg_list) == 4:
        X0_str, U_str, serv_ret_inter_str, sim_time_incr_str = msg_list

        try:
            X0 = list(X0_str.split(" "))
            X0 = [float(i) for i in X0]
        except:
            print("ValueError: could not resolve X0")
            raise

        try:
            U = np.fromstring(U_str.split(",")[0], dtype=float, sep=" ")
            for str_array in U_str.split(",")[1:]:
                U = np.vstack((U,np.fromstring(str_array, dtype=float, sep=" ")))
        except:
            print("ValueError: could not resolve U")
            raise

        try:
            server_return_interval = float(serv_ret_inter_str)
        except:
            print("ValueError: could not resolve server_return_interval")
            raise

        try:
            sim_time_increment = float(sim_time_incr_str)
        except:
            print("ValueError: could not resolve sim_time_increment")
            raise

    return X0, U, server_return_interval, sim_time_increment


def encode_request_msg_to_txt(msg):
    txt_msg = ""
    for part in msg:
        dim = None
        if isinstance(part, list):
            try:
                dim = len(part[0])
            except:
                dim = 1
        elif isinstance(part, np.ndarray):
            dim = len(part.shape)
        elif isinstance(part, float) or isinstance(part, int):
            dim = 0
        if dim == 1:
            txt_msg += " ".join(str(i) for i in part)
            txt_msg += "\n"
        elif dim == 2:
            for item in part:
                txt_msg += " ".join(str(i) for i in item)
                txt_msg += ","
            txt_msg = txt_msg[:-1]
            txt_msg += "\n"
        elif dim == 0:
            txt_msg += str(part)
            txt_msg += "\n"
        else:
            print("ValueError")
    txt_msg = txt_msg[:-1]
    return txt_msg


def decode_answer_msg_from_txt(msg):
    msg_list = msg.split(",")
    # if len(msg_list) < 3:
    #     X1 = np.fromstring(msg.split(",")[0], dtype=float, sep=" ")
    # else:
    if len(msg_list[0].split(" ")) == 7 or len(msg_list[0].split(" ")) == 10:
        try:
            X1 = np.fromstring(msg.split(",")[0], dtype=float, sep=" ")
            for str_array in msg.split(",")[1:]:
                X1 = np.vstack((X1, np.fromstring(str_array, dtype=float, sep=" ")))
        except:
            print("ValueError: could not resolve X1")
            raise
    else:
        X1 = msg
    return X1


def encode_answer_msg_to_txt(msg):
    txt_msg = ""
    if isinstance(msg, np.ndarray):
        # if len(msg.shape) < 2:
        #     try:
        #         txt_msg += " ".join(str(i) for i in msg)
        #         # txt_msg += ","
        #     except:
        #         print("ValueError")
        #         raise
        # else:
        try:
            for item in msg:
                txt_msg += " ".join(str(i) for i in item)
                txt_msg += ","
            # print('before', txt_msg)
            txt_msg = txt_msg[:-1]
            # print('after', txt_msg)
        except:
            print("ValueError")
            raise
    else:
        txt_msg = str(msg)
    return txt_msg