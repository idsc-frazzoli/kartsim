#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 24.05.19 09:56

@author: mvb
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:11:42 2019

@author: mvb
"""

from multiprocessing.connection import Client
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt


def main():
    #___user inputs

    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190902/20190902T174135_05/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190905/20190905T191253_06/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190909/20190909T174744_07/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190912/20190912T162356_05/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190916/20190916T175046_05/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190921/20190921T124329_10/mpc'
    # pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190923/20190923T161636_03/mpc'
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190926/20190926T121623_05/mpc'
    mpcsolfiles = []
    for r, d, f in os.walk(pathmpcsolutiondata):
        for file in f:
            if '.csv' in file:
                mpcsolfiles.append([os.path.join(r, file),file])
    mpcsolfiles.sort()

    part = 80
    # for file_path, file_name in mpcsolfiles[part:part+1]:
    # for file_path, file_name in mpcsolfiles[245:246]:
    # for file_path, file_name in mpcsolfiles[60:100]:

    solve_times = []
    t0 = 0
    vx_ref = []
    vy_ref = []
    vtheta_ref = []
    BETA_ref = []
    AB_ref = []
    TV_ref = []
    t_offset = 0.0
    t_abs0 = None
    # for file_path, file_name in mpcsolfiles[130:150]:
    # for file_path, file_name in mpcsolfiles[100:150]:
    # for file_path, file_name in mpcsolfiles[220:280:5]:
    for file_path, file_name in mpcsolfiles[:]:
        if t_abs0 is None:
            mpc_sol_data = pd.read_csv(str(mpcsolfiles[0][0]), header=None,
                                       names=["U wheel left", "U wheel right", "U dotS", "U brake", "X U AB", "time",
                                              "X Ux", "X Uy", "X dotPsi", "X X", "X Y", "X Psi", "X w2L", "X w2R",
                                              "X s", "X bTemp"])
            t_abs0 = mpc_sol_data['time'][0]
        print(f'Loading file {file_name}')
        try:
            mpc_sol_data = pd.read_csv(str(file_path), header=None,
                                       names=["U wheel left", "U wheel right", "U dotS", "U brake", "X U AB", "time",
                                              "X Ux", "X Uy", "X dotPsi", "X X", "X Y", "X Psi", "X w2L", "X w2R",
                                              "X s", "X bTemp"])

        except:
            print('Could not open file at', file_path)
            raise
        # print(mpc_sol_data.head())
        # print(type(mpc_sol_data))


        #___simulation parameters
        # data_time_step = np.round(mpc_sol_data['time'].iloc[1] - mpc_sol_data['time'].iloc[0],3)  # [s] Log data sampling time step
        # sim_time_increment = data_time_step    # [s] time increment used in integration scheme inside simulation server (time step for logged simulation data)
        # simTime = np.round(mpc_sol_data['time'].iloc[-1] - mpc_sol_data['time'].iloc[0])  # [s] Total simulation time
        # simTime = data_time_step
        # initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]

        if t0 != 0:
            solve_times.append(mpc_sol_data['time'][0] - t0)
        t0 = mpc_sol_data['time'][0]


        # AB = (mpc_sol_data['U wheel left'] + mpc_sol_data['U wheel right']) / 2.0
        # TV = (mpc_sol_data['U wheel right'] - mpc_sol_data['U wheel left']) / 2.0
        # steerCal = mpc_sol_data['X s']
        # BETA = -0.63 * np.power(steerCal, 3) + 0.94 * steerCal
        #
        # U = np.array((mpc_sol_data['time'].values-t_abs0+t_offset,
        #               BETA.values,
        #               AB.values,
        #               TV.values))
        #
        # Y = np.array((mpc_sol_data['time']-t_abs0+t_offset,
        #               np.add(mpc_sol_data['X X'], np.cos(mpc_sol_data['X Psi']) * 0.46),
        #               np.add(mpc_sol_data['X Y'], np.sin(mpc_sol_data['X Psi']) * 0.46),
        #               mpc_sol_data['X Psi'],
        #               mpc_sol_data['X Ux'],
        #               np.add(mpc_sol_data['X Uy'],mpc_sol_data['X dotPsi'][0]*0.46),
        #               # mpc_sol_data['X Uy']+0.88,
        #               mpc_sol_data['X dotPsi'])).transpose()
        # # ______^^^______
        #
        # arrow_length = 1
        #
        # plt.figure(1)
        #
        # plt.plot(Y[:,1],Y[:,2],'r', linewidth=0.5)
        # plt.scatter(Y[0,1],Y[0,2],c='r')
        # # plt.plot(X1[:, 1], X1[:, 2], 'b')
        # # plt.scatter(X1[0, 1], X1[0, 2], c='b')
        # for i in range(len(Y[:,1])):
        #     plt.arrow(Y[i,1],Y[i,2], arrow_length * np.cos(Y[i, 3]), arrow_length * np.sin(Y[i, 3]),color = 'm', linewidth=0.5, alpha=0.5)
        # # for i in range(len(Y[:,1])):
        # #     plt.arrow(Y[i,1],Y[i,2], arrow_length * np.cos(Y[i, 3]+BETA[i]), arrow_length * np.sin(Y[i, 3]+BETA[i]),color='m')
        # # plt.plot(Y[:, 0], Y[:, 1], 'r')
        # # plt.plot(Y[:, 0], Y[:, 2], 'r')
        # # plt.axis('equal')
        # plt.legend(['MPC prediction','Kartsim (RK45)'])
        # plt.xlabel('pose x')
        # plt.ylabel('pose y')
        # plt.axis('equal')
        # # plt.title('Euler Integration')
        # # plt.hold()
        #
        # # plt.figure(2)
        # # # plt.plot(Y[:,0], Y[:,3], 'r')
        # # # plt.plot(X1[:,0],X1[:,3], 'b')
        # # # plt.plot( Y[:, -1], 'r')
        # # # plt.plot( X1[:, -1], 'b')
        # # plt.plot(Y[:, 0], Y[:, 6], 'r')
        # # plt.plot(X1[:, 0], X1[:, 6], 'b')
        # # plt.plot(Y[:, 0], Y[:, 3], 'r')
        # # plt.plot(X1[:, 0], X1[:, 3], 'b')
        # #
        #
        # plt.figure(3)
        # plt.plot(Y[:, 0], Y[:, 6], 'g', linewidth=0.5, alpha=0.5)
        # plt.plot(Y[:, 0], Y[:, 5], 'r', linewidth=0.5, alpha=0.5)
        # plt.plot(Y[:, 0], Y[:, 4], 'b', linewidth=0.5, alpha=0.5)
        # plt.scatter(Y[0, 0], Y[0, 6], c='g')
        # plt.scatter(Y[0, 0], Y[0, 5], c='r')
        # plt.scatter(Y[0, 0], Y[0, 4], c='b')
        # # plt.plot(Y[1:, 0], x_dot, 'c')
        # # plt.plot(Y[1:, 0], y_dot, 'c')
        # # plt.plot(Y[:, 0], np.sqrt(np.square(Y[:, 4]) + np.square(Y[:, 5])), 'k')
        # plt.legend(['dottheta', 'vy', 'vx'])
        #
        #
        # # plt.figure(4)
        # # plt.plot(Y[:,0], Y_slip_angle, 'r')
        # # plt.plot(X1[:,0], X1_slip_angle, 'orange')
        # # plt.plot(Y[1:,0], Y_slip_angle_from_pose, 'm')
        # # plt.plot(X1[1:,0], X1_slip_angle_from_pose, 'c')
        # # plt.title('slip angle [rad]')
        # #
        # # plt.figure(5)
        # # plt.plot(Y[1:, 0]-0.05, vy, 'm')
        # # # plt.scatter(Y[1:, 0], vy, c='m')
        # # plt.plot(Y[:, 0], Y[:, 5], 'r')
        # # # plt.scatter(Y[0, 0], Y[0, 5], c='r')
        # # plt.plot(Y[:, 0], Y[:, 4], 'r')
        # # # plt.scatter(Y[0, 0], Y[0, 4], c='orange')
        # # plt.plot(Y[:-1, 0], vx, 'c')
        # # # plt.plot(Y[:, 0], Y[:, 4], 'b')
        # # plt.legend(['MPC from pose','MPC output','Kartsim output', 'Kartsim from pose'])
        # # plt.xlabel('time [s]')
        # # plt.ylabel('U y [m/s]')
        #
        # plt.figure(6)
        # plt.plot(U[0,:], U[1,:],c='m', linewidth=0.5, alpha=0.5)
        # plt.plot(U[0,:], U[2,:],c='b', linewidth=0.5, alpha=0.5)
        # plt.plot(U[0,:], U[3,:],c='g', linewidth=0.5, alpha=0.5)
        # plt.scatter(U[0, 0], U[1, 0], c='m')
        # plt.scatter(U[0, 0], U[2, 0], c='b')
        # plt.scatter(U[0, 0], U[3, 0], c='g')
        # plt.legend(['BETA','AB','TV'])
        # plt.grid('on')
        #
        # vx_ref.append([Y[0, 0], Y[0, 4]])
        # vy_ref.append([Y[0, 0], Y[0, 5]])
        # vtheta_ref.append([Y[0, 0], Y[0, 6]])
        # BETA_ref.append([U[0, 0], U[1, 0]])
        # AB_ref.append([U[0, 0], U[2, 0]])
        # TV_ref.append([U[0, 0], U[3, 0]])

    sns.distplot(solve_times)
    plt.title('MPC normal - 0s delay')
    print('solve times:', solve_times)
    print('avg solve time:', np.average(solve_times))
    print('median solve time:', np.median(solve_times))
    print('std solve time:', np.std(solve_times))

    # vx_ref = np.array(vx_ref)
    # vy_ref = np.array(vy_ref)
    # vtheta_ref = np.array(vtheta_ref)
    # BETA_ref = np.array(BETA_ref)
    # AB_ref = np.array(AB_ref)
    # TV_ref = np.array(TV_ref)
    #
    # plt.figure(3)
    # plt.plot(vx_ref[:,0], vx_ref[:,1], 'b')
    # plt.plot(vy_ref[:,0], vy_ref[:,1], 'r')
    # plt.plot(vtheta_ref[:,0], vtheta_ref[:,1], 'g')
    # # plt.legend(['vy mpc', 'vy kartsim', 'vy reference'])
    #
    # plt.figure(6)
    # plt.plot(BETA_ref[:,0], BETA_ref[:,1], 'm')
    # plt.plot(AB_ref[:,0], AB_ref[:,1], 'b')
    # plt.plot(TV_ref[:,0], TV_ref[:,1], 'g')
    #
    # plt.show()
    print("Done.")


def getpreprodata(pathpreprodata):
    files = []
    for r, d, f in os.walk(pathpreprodata):
        for file in f:
            if '.pkl' in file:
                files.append(os.path.join(r, file))
    for filePath in files[0:1]:
        try:
            with open(filePath, 'rb') as f:
                mpc_sol_data = pickle.load(f)
        except:
            print('Could not open file at', filePath)
            mpc_sol_data = pd.DataFrame()

    return mpc_sol_data

if __name__ == '__main__':
    main()