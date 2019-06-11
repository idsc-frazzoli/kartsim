#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import numpy as np
import os
from dataanalysisV2.data_io import getPKL, getDirectories


def main():

    pathrootsimdata = '/home/mvb/0_ETH/01_MasterThesis/SimData'
    simfolders = getDirectories(pathrootsimdata)
    simfolders.sort()
    defaultsim = simfolders[-1]
    pathsimdata = pathrootsimdata + '/' + defaultsim
    print('Creating reference signals... Reading data from ', pathsimdata)
    evaluationhorizon = 1      #[s] time horizon for which simulation runs until initial conditions are reset to values from log data

    csvfiles = []
    pklfiles = []
    for r, d, f in os.walk(pathsimdata):
        for file in f:
            if '.csv' in file:
                csvfiles.append([os.path.join(r, file), file])
            if '.pkl' in file:
                pklfiles.append([os.path.join(r, file), file])
    for index in range(len(pklfiles)):
        try:
            rawdataframe = getPKL(pklfiles[index][0])
        except:
            print('EmptyDataError: could not read data from file ', pklfiles[index][1])
            raise
        dt = np.round(rawdataframe['time [s]'].diff().mean(),4)
        vx = 0
        vy = 0
        vtheta = 0
        evalrefdata = rawdataframe[
            ['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]',
             'pose vtheta [rad*s^-1]', 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]',
             'pose atheta [rad*s^-2]', 'steer position cal [n.a.]', 'brake position effective [m]',
             'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]']].copy(deep=True)
                                            # 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]', 'MH BETA [rad]', 'MH AB [m*s^-2]', 'MH TV [rad*s^-2]']].copy(deep=True)
        for i in range(len(rawdataframe)):
            if i * dt % evaluationhorizon < dt:
                vxRel = rawdataframe['vehicle vx [m*s^-1]'][i]
                vyRel = rawdataframe['vehicle vy [m*s^-1]'][i]
                vtheta = rawdataframe['pose vtheta [rad*s^-1]'][i]
                if i > 0:
                    theta = evalrefdata['pose theta [rad]'][i]
                    vx = vxRel * np.cos(theta) - vyRel * np.sin(theta)
                    vy = vyRel * np.cos(theta) + vxRel * np.sin(theta)
                    evalrefdata['pose x [m]'][i] = rawdataframe['pose x [m]'][i]
                    evalrefdata['pose y [m]'][i] = rawdataframe['pose y [m]'][i]
                    evalrefdata['pose theta [rad]'][i] = rawdataframe['pose theta [rad]'][i]
                    evalrefdata['vehicle vx [m*s^-1]'][i] = vxRel
                    evalrefdata['vehicle vy [m*s^-1]'][i] = vyRel
                    evalrefdata['pose vtheta [rad*s^-1]'][i] = vtheta
                    evalrefdata['vehicle ax local [m*s^-2]'][i] = 0
                    evalrefdata['vehicle ay local [m*s^-2]'][i] = 0
                    evalrefdata['pose atheta [rad*s^-2]'][i] = 0
                continue
            if i > 0:
                theta = evalrefdata['pose theta [rad]'][i]
                vx = vxRel * np.cos(theta) - vyRel * np.sin(theta)
                vy = vyRel * np.cos(theta) + vxRel * np.sin(theta)
                evalrefdata['pose x [m]'][i] = evalrefdata['pose x [m]'][i-1] + vx * dt
                evalrefdata['pose y [m]'][i] = evalrefdata['pose y [m]'][i-1] + vy * dt
                evalrefdata['pose theta [rad]'][i] = evalrefdata['pose theta [rad]'][i-1] + vtheta * dt
                evalrefdata['vehicle vx [m*s^-1]'][i] = vxRel
                evalrefdata['vehicle vy [m*s^-1]'][i] = vyRel
                evalrefdata['pose vtheta [rad*s^-1]'][i] = vtheta
                evalrefdata['vehicle ax local [m*s^-2]'][i] = 0
                evalrefdata['vehicle ay local [m*s^-2]'][i] = 0
                evalrefdata['pose atheta [rad*s^-2]'][i] = 0

        savePathName = pathsimdata + '/' + pklfiles[index][1][:-19] + '_evaluationreference.csv'

        evalrefdata.to_csv(savePathName, index=False,
                         header=['time [s]', 'pose x [m]', 'pose y [m]', 'pose theta [rad]', 'vehicle vx [m*s^-1]', 'vehicle vy [m*s^-1]', 'pose vtheta [rad*s^-1]',
                                 'vehicle ax local [m*s^-2]', 'vehicle ay local [m*s^-2]', 'pose atheta [rad*s^-2]', 'steer position cal [n.a.]', 'brake position effective [m]',
             'motor torque cmd left [A_rms]', 'motor torque cmd right [A_rms]'])
        # print('Evaluation reference saved to ', savePathName)

if __name__ == '__main__':
    main()
