#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import numpy as np
import os
from dataanalysisV2.dataIO import getPKL, getDirectories


def main():

    pathrootsimdata = '/home/mvb/0_ETH/01_MasterThesis/SimData'
    simfolders = getDirectories(pathrootsimdata)
    simfolders.sort()
    defaultsim = simfolders[-1]
    pathsimdata = pathrootsimdata + '/' + defaultsim
    print('Reading data from ', pathsimdata)
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
        dt = np.round(rawdataframe['time'].diff().mean(),4)
        vx = 0
        vy = 0
        vtheta = 0
        evalrefdata = rawdataframe[['time', 'pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta',
                                 'MH BETA', 'MH AB', 'MH TV']].copy(deep=True)
        for i in range(len(rawdataframe)):
        # for i in range(350):
            if i * dt % evaluationhorizon < dt:
                vxRel = rawdataframe['vehicle vx'][i]
                vyRel = rawdataframe['vehicle vy'][i]
                vtheta = rawdataframe['pose vtheta'][i]
                if i > 0:
                    theta = evalrefdata['pose theta'][i]
                    vx = vxRel * np.cos(theta) - vyRel * np.sin(theta)
                    vy = vyRel * np.cos(theta) + vxRel * np.sin(theta)
                    evalrefdata['pose x'][i] = rawdataframe['pose x'][i]
                    evalrefdata['pose y'][i] = rawdataframe['pose y'][i]
                    evalrefdata['pose theta'][i] = rawdataframe['pose theta'][i]
                    evalrefdata['vehicle vx'][i] = vxRel
                    evalrefdata['vehicle vy'][i] = vyRel
                    evalrefdata['pose vtheta'][i] = vtheta
                continue
            if i > 0:
                theta = evalrefdata['pose theta'][i]
                vx = vxRel * np.cos(theta) - vyRel * np.sin(theta)
                vy = vyRel * np.cos(theta) + vxRel * np.sin(theta)
                evalrefdata['pose x'][i] = evalrefdata['pose x'][i-1] + vx * dt
                evalrefdata['pose y'][i] = evalrefdata['pose y'][i-1] + vy * dt
                evalrefdata['pose theta'][i] = evalrefdata['pose theta'][i-1] + vtheta * dt
                evalrefdata['vehicle vx'][i] = vxRel
                evalrefdata['vehicle vy'][i] = vyRel
                evalrefdata['pose vtheta'][i] = vtheta

        savePathName = pathsimdata + '/' + pklfiles[index][1][:-19] + '_evaluationreference.csv'

        evalrefdata.to_csv(savePathName, index=False,
                         header=['time', 'pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta',
                                 'MH BETA', 'MH AB', 'MH TV'])
        print('Evaluation reference saved to ', savePathName)

if __name__ == '__main__':
    main()
