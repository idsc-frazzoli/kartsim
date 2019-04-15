#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import numpy as np
import os
import dataanalysis.pyqtgraph.dataIO as dIO


def main():

    pathrootsimdata = '/home/mvb/0_ETH/01_MasterThesis/SimData'
    simfolders = dIO.getDirectories(pathrootsimdata)
    simfolders.sort()
    defaultsim = simfolders[-1]
    pathsimdata = pathrootsimdata + '/' + defaultsim

    csvfiles = []
    pklfiles = []
    for r, d, f in os.walk(pathsimdata):
        for file in f:
            if '.csv' in file:
                csvfiles.append([os.path.join(r, file), file])
            if '.pkl' in file:
                pklfiles.append([os.path.join(r, file), file])

    try:
        rawdataframe = dIO.getPKL(pklfiles[0][0])
    except:
        print('EmptyDataError: could not read data from file ', pklfiles[0][1])
        raise

    for index in range(len(csvfiles)):
        try:
            simdataframe = dIO.getCSV(csvfiles[index][0])
        except:
            print('EmptyDataError: could not read data from file ', csvfiles[index][1])
            raise

        start = 0
        part = -1
        rawvaldata = rawdataframe[['pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta']]
        simvaldata = simdataframe[['pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta']]
        diff_pose = rawvaldata[start:start+part] - simvaldata[start:start+part]
        # print(np.square(diff))
        rmse = np.sqrt(np.sum(np.square(diff_pose)) / len(rawdataframe[start:start+part]))
        print('Evaluation for ', csvfiles[index][1])
        print(rmse)
        print('Overall score: ', rmse.sum(),'\n')


if __name__ == '__main__':
    main()
