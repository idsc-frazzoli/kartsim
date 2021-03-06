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

from simulator.textcommunication import encode_request_msg_to_txt, decode_answer_msg_from_txt


def main():
    #___user inputs

    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/simtompc_comparison' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/simtompc_comparison_19082019' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190827/20190827T175738_03/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190827/20190827T175738_05/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190827/20190827T170622_01/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190827/20190827T170622_02/mpc' #path where all the raw, sorted data is that you want to sample and or batch and or split
    pathmpcsolutiondata = '/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/dynamics_newFormat/cuts/20190905/20190905T191253_06/mpc'
    mpcsolfiles = []
    for r, d, f in os.walk(pathmpcsolutiondata):
        for file in f:
            if '.csv' in file:
                mpcsolfiles.append([os.path.join(r, file),file])
    mpcsolfiles.sort()


    real_time = False
    real_time_factor = 1
    _wait_for_real_time = 0

    if real_time:
        server_return_interval = 0.1*real_time_factor  # [s] simulation time after which result is returned from server
        _wait_for_real_time = server_return_interval*0.9*(1.0/real_time_factor)

    connected = False
    while not connected:
        try:
            address = ('localhost', 6000)
            conn = Client(address, authkey=b'kartSim2019')
            connected = True
        except ConnectionRefusedError:
            print('ConnectionRefusedError')
            pass

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
    t_offset = -3.9
    # for file_path, file_name in mpcsolfiles[130:150]:
    for file_path, file_name in mpcsolfiles[-50:]:
    # for file_path, file_name in mpcsolfiles[34:50:5]:

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
        data_time_step = np.round(mpc_sol_data['time'].iloc[1] - mpc_sol_data['time'].iloc[0],3)  # [s] Log data sampling time step
        sim_time_increment = data_time_step    # [s] time increment used in integration scheme inside simulation server (time step for logged simulation data)
        simTime = np.round(mpc_sol_data['time'].iloc[-1] - mpc_sol_data['time'].iloc[0])  # [s] Total simulation time
        # simTime = data_time_step
        # initial state [simulationTime, x, y, theta, vx, vy, vrot, beta, accRearAxle, tv]
        if t0 != 0:
            solve_times.append(mpc_sol_data['time'][0] - t0)
        t0 = mpc_sol_data['time'][0]

        X0 = [mpc_sol_data['time'][0]+t_offset,
              mpc_sol_data['X X'][0] + np.cos(mpc_sol_data['X Psi'][0]) * 0.46,
              mpc_sol_data['X Y'][0] + np.sin(mpc_sol_data['X Psi'][0]) * 0.46,
              mpc_sol_data['X Psi'][0],
              mpc_sol_data['X Ux'][0],
              np.add(mpc_sol_data['X Uy'][0],mpc_sol_data['X dotPsi'][0]*0.46),
              mpc_sol_data['X dotPsi'][0]]

        AB = (mpc_sol_data['U wheel left'] + mpc_sol_data['U wheel right']) / 2.0
        TV = (mpc_sol_data['U wheel right'] - mpc_sol_data['U wheel left']) / 2.0
        steerCal = mpc_sol_data['X s']
        BETA = -0.63 * np.power(steerCal, 3) + 0.94 * steerCal

        U = np.array((mpc_sol_data['time'].values+t_offset,
                      BETA.values,
                      AB.values,
                      TV.values))

        Y = np.array((mpc_sol_data['time']+t_offset,
                      np.add(mpc_sol_data['X X'], np.cos(mpc_sol_data['X Psi']) * 0.46),
                      np.add(mpc_sol_data['X Y'], np.sin(mpc_sol_data['X Psi']) * 0.46),
                      mpc_sol_data['X Psi'],
                      mpc_sol_data['X Ux'],
                      np.add(mpc_sol_data['X Uy'],mpc_sol_data['X dotPsi'][0]*0.46),
                      # mpc_sol_data['X Uy']+0.88,
                      mpc_sol_data['X dotPsi'])).transpose()
        # ______^^^______

        print('Simulation with file ', file_name)

        txt_msg = encode_request_msg_to_txt([X0, U, simTime, sim_time_increment])
        # conn.send([X0, U, server_return_interval, sim_time_increment])
        conn.send(txt_msg)

        answer_msg = conn.recv()
        X1 = decode_answer_msg_from_txt(answer_msg)
        X1 = X1[:-1,:]

        arrow_length = 1

        plt.figure(1)

        plt.plot(Y[:,1],Y[:,2],'r', linewidth=0.5)
        plt.scatter(Y[0,1],Y[0,2],c='r')
        # plt.plot(X1[:, 1], X1[:, 2], 'b')
        # plt.scatter(X1[0, 1], X1[0, 2], c='b')
        for i in range(len(Y[:,1])):
            plt.arrow(Y[i,1],Y[i,2], arrow_length * np.cos(Y[i, 3]), arrow_length * np.sin(Y[i, 3]),color = 'm', linewidth=0.5, alpha=0.5)
        # for i in range(len(Y[:,1])):
        #     plt.arrow(Y[i,1],Y[i,2], arrow_length * np.cos(Y[i, 3]+BETA[i]), arrow_length * np.sin(Y[i, 3]+BETA[i]),color='m')
        # for i in range(len(X1[:, 1])):
        #     plt.arrow(X1[i, 1], X1[i, 2], arrow_length * np.cos(X1[i, 3]), arrow_length * np.sin(X1[i, 3]),color='c')
        # for i in range(len(X1[:,1])):
        #     plt.arrow(X1[i,1],X1[i,2], arrow_length * np.cos(X1[i, 3]+BETA[i]), arrow_length * np.sin(X1[i, 3]+BETA[i]),color='m')
        # plt.plot(Y[:, 0], Y[:, 1], 'r')
        # plt.plot(Y[:, 0], Y[:, 2], 'r')
        # plt.plot(X1[:, 0], X1[:, 1], 'b')
        # plt.plot(X1[:, 0], X1[:, 2], 'b')
        # plt.axis('equal')
        plt.legend(['MPC prediction','Kartsim (RK45)'])
        plt.xlabel('pose x')
        plt.ylabel('pose y')
        plt.axis('equal')
        # plt.title('Euler Integration')
        plt.hold

        # plt.figure(2)
        # # plt.plot(Y[:,0], Y[:,3], 'r')
        # # plt.plot(X1[:,0],X1[:,3], 'b')
        # # plt.plot( Y[:, -1], 'r')
        # # plt.plot( X1[:, -1], 'b')
        # plt.plot(Y[:, 0], Y[:, 6], 'r')
        # plt.plot(X1[:, 0], X1[:, 6], 'b')
        # plt.plot(Y[:, 0], Y[:, 3], 'r')
        # plt.plot(X1[:, 0], X1[:, 3], 'b')
        #
        x_dot = (Y[1:, 1] - Y[:-1, 1]) / data_time_step
        y_dot = (Y[1:, 2] - Y[:-1, 2]) / data_time_step
        theta = (Y[1:, 3] + Y[:-1, 3]) / 2
        xx_dot = (X1[1:, 1] - X1[:-1, 1]) / sim_time_increment
        yx_dot = (X1[1:, 2] - X1[:-1, 2]) / sim_time_increment
        theta1 = (X1[1:, 3] + X1[:-1, 3]) / 2

        plt.figure(3)
        plt.plot(Y[:, 0], Y[:, 6], 'g', linewidth=0.5, alpha=0.5)
        plt.plot(Y[:, 0], Y[:, 5], 'r', linewidth=0.5, alpha=0.5)
        plt.plot(Y[:, 0], Y[:, 4], 'b', linewidth=0.5, alpha=0.5)
        plt.scatter(Y[0, 0], Y[0, 6], c='g')
        plt.scatter(Y[0, 0], Y[0, 5], c='r')
        plt.scatter(Y[0, 0], Y[0, 4], c='b')
        # plt.plot(Y[1:, 0], x_dot, 'c')
        # plt.plot(Y[1:, 0], y_dot, 'c')
        # plt.plot(X1[:, 0], X1[:, 5], 'orange', linewidth=0.5, alpha=0.5)
        # plt.plot(X1[:, 0], X1[:, 4], 'g', linewidth=0.5, alpha=0.5)
        # plt.plot(X1[1:, 0], xx_dot, 'y')
        # plt.plot(X1[1:, 0], yx_dot, 'y')
        # plt.plot(Y[:, 0], np.sqrt(np.square(Y[:, 4]) + np.square(Y[:, 5])), 'k')
        # plt.plot(X1[:, 0], np.sqrt(np.square(X1[:, 4]) + np.square(X1[:, 5])), 'k')
        plt.legend(['dottheta', 'vy', 'vx'])

        # # #
        vx = x_dot * np.cos(theta) + y_dot * np.sin(theta)
        vy = -x_dot * np.sin(theta) + y_dot * np.cos(theta)

        vxx = xx_dot * np.cos(theta1) + yx_dot * np.sin(theta1)
        vyx = -xx_dot * np.sin(theta1) + yx_dot * np.cos(theta1)
        #
        # Y_slip_angle = np.arctan(np.divide(Y[:, 5], Y[:, 4]))
        # X1_slip_angle = np.arctan(np.divide(X1[:, 5], X1[:, 4]))
        # Y_slip_angle_from_pose = np.arctan(np.divide(vy, vx))
        # X1_slip_angle_from_pose = np.arctan(np.divide(vyx, vxx))
        #
        # plt.figure(4)
        # plt.plot(Y[:,0], Y_slip_angle, 'r')
        # plt.plot(X1[:,0], X1_slip_angle, 'orange')
        # plt.plot(Y[1:,0], Y_slip_angle_from_pose, 'm')
        # plt.plot(X1[1:,0], X1_slip_angle_from_pose, 'c')
        # plt.title('slip angle [rad]')
        #
        # plt.figure(5)
        # plt.plot(Y[1:, 0]-0.05, vy, 'm')
        # # plt.scatter(Y[1:, 0], vy, c='m')
        # plt.plot(Y[:, 0], Y[:, 5], 'r')
        # # plt.scatter(Y[0, 0], Y[0, 5], c='r')
        # plt.plot(Y[:, 0], Y[:, 4], 'r')
        # # plt.scatter(Y[0, 0], Y[0, 4], c='orange')
        # plt.plot(X1[:, 0], X1[:, 5], 'orange')
        # # plt.scatter(X1[0, 0], X1[0, 5], c='orange')
        # plt.plot(X1[1:, 0]-0.05, vyx, 'c')
        # # plt.scatter(X1[1:, 0], vyx, c='c')
        # plt.plot(Y[:-1, 0], vx, 'c')
        # # plt.plot(Y[:, 0], Y[:, 4], 'b')
        # plt.plot(X1[:, 0], X1[:, 4], 'g')
        # plt.plot(X1[:-1, 0], vxx, c='y')
        # plt.legend(['MPC from pose','MPC output','Kartsim output', 'Kartsim from pose'])
        # plt.xlabel('time [s]')
        # plt.ylabel('U y [m/s]')

        plt.figure(6)
        plt.plot(U[0,:], U[1,:],c='m', linewidth=0.5, alpha=0.5)
        plt.plot(U[0,:], U[2,:],c='b', linewidth=0.5, alpha=0.5)
        plt.plot(U[0,:], U[3,:],c='g', linewidth=0.5, alpha=0.5)
        plt.scatter(U[0, 0], U[1, 0], c='m')
        plt.scatter(U[0, 0], U[2, 0], c='b')
        plt.scatter(U[0, 0], U[3, 0], c='g')
        plt.legend(['BETA','AB','TV'])
        plt.grid('on')

        vx_ref.append([Y[0, 0], Y[0, 4]])
        vy_ref.append([Y[0, 0], Y[0, 5]])
        vtheta_ref.append([Y[0, 0], Y[0, 6]])
        BETA_ref.append([U[0, 0], U[1, 0]])
        AB_ref.append([U[0, 0], U[2, 0]])
        TV_ref.append([U[0, 0], U[3, 0]])

    print('solve times:', solve_times)
    print('avg solve time:', np.average(solve_times))

    vx_ref = np.array(vx_ref)
    vy_ref = np.array(vy_ref)
    vtheta_ref = np.array(vtheta_ref)
    BETA_ref = np.array(BETA_ref)
    AB_ref = np.array(AB_ref)
    TV_ref = np.array(TV_ref)

    plt.figure(3)
    plt.plot(vx_ref[:,0], vx_ref[:,1], 'b')
    plt.plot(vy_ref[:,0], vy_ref[:,1], 'r')
    plt.plot(vtheta_ref[:,0], vtheta_ref[:,1], 'g')
    # plt.legend(['vy mpc', 'vy kartsim', 'vy reference'])

    plt.figure(6)
    plt.plot(BETA_ref[:,0], BETA_ref[:,1], 'm')
    plt.plot(AB_ref[:,0], AB_ref[:,1], 'b')
    plt.plot(TV_ref[:,0], TV_ref[:,1], 'g')

    plt.show()
    print("Done.")
    #
    #     # generate simulation info file and store it in target folder
    #     try:
    #         _ = open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'r')
    #         os.remove(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt')
    #         with open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'a') as the_file:
    #             if validation:
    #                 the_file.write('simulation mode:                    evaluation' + '\n')
    #                 the_file.write('initial condition reset interval:   ' + str(validationhorizon) + 's\n')
    #             else:
    #                 the_file.write('simulation mode:                    normal simulation' + '\n')
    #             the_file.write('source folder:                      ' + pathpreprodata + '\n')
    #             the_file.write('simulation time step:               ' + str(server_return_interval) + 's\n')
    #             the_file.write('total simulation time:              ' + str(simTime) + 's\n')
    #             the_file.write('time step in data:                  ' + str(data_time_step) + 's\n')
    #             the_file.write('initial conditions:                 ' + str(X0[0]) + '\n')
    #             for item in X0[1:]:
    #                 the_file.write('                                    ' + str(item) + '\n')
    #     except FileNotFoundError:
    #         with open(pathsavedata + '/' + fileName[:-12] + '_simulationinfo.txt', 'a') as the_file:
    #             if validation:
    #                 the_file.write('simulation mode:                    evaluation' + '\n')
    #                 the_file.write('initial condition reset interval:   ' + str(validationhorizon) + 's\n')
    #             else:
    #                 the_file.write('simulation mode:                    normal simulation' + '\n')
    #             the_file.write('source folder:                      ' + pathpreprodata + '\n')
    #             the_file.write('simulation time step:               ' + str(server_return_interval) + 's\n')
    #             the_file.write('total simulation time:              ' + str(simTime) + 's\n')
    #             the_file.write('time step in data:                  ' + str(data_time_step) + 's\n')
    #             the_file.write('initial conditions:                 ' + str(X0[0]) + '\n')
    #             for item in X0[1:]:
    #                 the_file.write('                                    ' + str(item) + '\n')
    # print('connection closed')
    # conn.close()
    # time.sleep(2)
    # print('Creating reference signal for evaluation...')
    # evalRef.main()
    # print('Evaluating results...')
    # evalCalc.main()
    # print('Evaluation complete!')



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