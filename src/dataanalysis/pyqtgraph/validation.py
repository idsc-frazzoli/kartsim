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

        try:
            fh = open(pathsimdata +'/evaluationresults.txt', 'r')
            if index == 0:
                os.remove(pathsimdata +'/evaluationresults.txt')
            with open(pathsimdata + '/evaluationresults.txt', 'a') as the_file:
                the_file.write('_____________________________________\n')
                the_file.write('Evaluation for ' + csvfiles[index][1] + '\n')
                the_file.write('Parameter      RMSE' + '\n')
                the_file.write(str(rmse.to_string()) + '\n')
                the_file.write('Overall score: ' + str(rmse.sum()) + '\n\n')
            print('Results saved to file', pathsimdata +'/evaluationresults.txt')
        except FileNotFoundError:
            with open(pathsimdata + '/evaluationresults.txt', 'a') as the_file:
                the_file.write('_____________________________________\n')
                the_file.write('Evaluation for ' + csvfiles[index][1] + '\n')
                the_file.write('Signal         RMSE' + '\n')
                the_file.write(str(rmse.to_string()) + '\n')
                the_file.write('Overall score: ' + str(rmse.sum()) + '\n\n')
            print('Results saved to file', pathsimdata + '/evaluationresults.txt')
            # Keep preset values


        print('Evaluation for ', csvfiles[index][1])
        print(rmse)
        print('Overall score: ', rmse.sum(),'\n')


if __name__ == '__main__':
    main()
