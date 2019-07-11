#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:04 2019

@author: mvb
"""
import numpy as np
import os
import dataIO as dIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    pathrootsimdata = '/home/mvb/0_ETH/01_MasterThesis/SimData'
    simfolders = dIO.getDirectories(pathrootsimdata)
    simfolders.sort()
    defaultsim = simfolders[-1]
    simfolder = defaultsim
    pathsimdata = pathrootsimdata + '/' + simfolder

    csvfiles = []
    pklfiles = []
    for r, d, f in os.walk(pathsimdata):
        for file in f:
            if '.csv' in file:
                csvfiles.append([os.path.join(r, file), file])
            if '.pkl' in file:
                pklfiles.append([os.path.join(r, file), file])
    csvfiles.sort()
    pklfiles.sort()
    simfiles = []
    for i in range(len(pklfiles)):
        simname = pklfiles[i][1][:18]
        simfiles.append([pklfiles[i]])
        for j in range(len(csvfiles)):
            if csvfiles[j][1][:18] == simname:
                simfiles[i].append(csvfiles[j])
    print(simfiles[0][0][0])

    for index in range(len(simfiles)):
        try:
            rawdataframe = dIO.getPKL(simfiles[index][0][0])
        except:
            print('FileNotFoundError: could not read data from file ', simfiles[index][0][0])
            raise


        rawvaldata = []
        simvaldata = []
        diff = []
        rmse = []
        for j in range(len(simfiles[index][1:])):
            try:
                simdataframe = dIO.getCSV(simfiles[index][j+1][0])
            except:
                print('FileNotFoundError: could not read data from file ', simfiles[index][j+1][0])
                raise

            start = 0
            part = -1
            rawvaldata.append(rawdataframe[['pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta']])
            simvaldata.append(simdataframe[['pose x', 'pose y', 'pose theta', 'vehicle vx', 'vehicle vy', 'pose vtheta']])
            diff.append(rawvaldata[j][start:start+part] - simvaldata[j][start:start+part])
            # print(np.square(diff))
            rmse.append(np.sqrt(np.sum(np.square(diff[j])) / len(rawdataframe[start:start + part])))

            print('Evaluation for ', simfiles[index][j+1][1], 'vs', simfiles[index][0][1])
            print(rmse[j])
            print('Overall score: ', np.sum(rmse[j]), '\n')

        #generating results page
        pdffilepath = pathsimdata + '/' + simfiles[index][0][1][:-19] + '_evaluation_plots.pdf'
        pdf = PdfPages(pdffilepath)
        infotext1 = simfolder + '\n'
        for k in range(len(simfiles[index][1:])):
            if k == 0:
                infotext2 = '\n\n\nRMSE\n'
            else:
                infotext2 += '\n\nRMSE\n'
            infotext1 += '_____________________________________\n'
            infotext1 +='Evaluation for ' + simfiles[index][k+1][1] + ' vs ' + simfiles[index][0][1] + '\n'
            infotext1 += 'Parameter' + '\n'
            for i in range(len(rmse[k])):
                infotext1 += str(rmse[k].index[i]) + '\n'
                infotext2 += str(rmse[k].values[i]) + '\n'
            infotext1 +='Overall score: ' + '\n\n'
            infotext2 += str(rmse[k].sum()) + '\n\n'
        resultsPage = plt.figure(figsize=(11.69, 8.27))
        resultsPage.clf()
        resultsPage.text(0.05, 0.95, infotext1,  size=12, ha="left", va='top')
        resultsPage.text(0.2, 0.95, infotext2,  size=12, ha="left", va='top')
        pdf.savefig()
        plt.close()
        # print('Evaluation for ' + simfiles[index][k+1][1] + ' vs ' + simfiles[index][0][1] + '\n')
        # print(rmse)


        # generating plots
        for debugtopic in rawvaldata[0].columns:
            for k in range(len(simfiles[index][1:])):
                # debugtopic = 'pose vtheta'
                savetopic = debugtopic.replace(' ', '_')
                fig, axs = plt.subplots(2, 1, figsize = (10,6))
                fig.suptitle(debugtopic + '\n' + csvfiles[index*2+k][1][:-4],fontsize = 16)
                # plt.figure(figsize=(10,10))
                axs[0].plot(rawvaldata[k][debugtopic][start:start+part], label = 'log data')
                axs[0].plot(simvaldata[k][debugtopic][start:start+part], label = 'simulation data')
                axs[0].legend()
                axs[1].plot(diff[k][debugtopic], color = 'r', label = 'error')
                axs[1].legend()
                pdf.savefig()
                plt.close()
                # plt.savefig(pathsimdata + '/' + savetopic + '-' + csvfiles[index * 2 + j][1][:-4] + '.pdf')
                # plt.show()


        #results page as txt file
        # resultsfilepath = pathsimdata +'/evaluationresults.txt'
        # for k in range(2):
        #     try:
        #         fh = open(resultsfilepath, 'r')
        #         fh.close()
        #         if index == 0 and k == 0:
        #             os.remove(resultsfilepath)
        #         with open(pathsimdata + '/evaluationresults.txt', 'a') as the_file:
        #             the_file.write('_____________________________________\n')
        #             the_file.write('Evaluation for ' + csvfiles[index*2+j][1] + ' vs ' + simfiles[index][0][1] + '\n')
        #             the_file.write('Parameter      RMSE' + '\n')
        #             the_file.write(str(rmse[k].to_string()) + '\n')
        #             the_file.write('Overall score: ' + str(rmse[k].sum()) + '\n\n')
        #     except FileNotFoundError:
        #         with open(pathsimdata + '/evaluationresults.txt', 'a') as the_file:
        #             the_file.write('_____________________________________\n')
        #             the_file.write('Evaluation for ' + csvfiles[index*2+j][1] + ' vs ' + simfiles[index][0][1] + '\n')
        #             the_file.write('Signal         RMSE' + '\n')
        #             the_file.write(str(rmse[k].to_string()) + '\n')
        #             the_file.write('Overall score: ' + str(rmse[k].sum()) + '\n\n')
        # print('Results saved to file', resultsfilepath)
        # # Keep preset values


        pdf.close()


if __name__ == '__main__':
    main()
