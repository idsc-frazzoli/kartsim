#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 24.10.19 15:24

@author: mvb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch

def main():
    data = [[['dynamic model (M.H.)',
              'dynamic residual model (0x0 MLP sym)',
              'dynamic residual model (1x16 MLP)',
              'dynamic residual model (2x16 MLP)',
              'kinematic residual model (1x16 MLP sym)',
              'kinematic residual model (1x24 MLP sym)',
              'kinematic residual model (1x32 MLP)',
              'kinematic residual model (2x16 MLP)',
              'black-box model (1x16 MLP sym)',
              'black-box model (1x24 MLP)',
              'black-box model (1x32 MLP)'
              ],[0.055,0.057,0.091,0.105,0.072,0.109,0.101,0.09,0.0866, 0.085, 0.099],
             ''],
            # 'Solve times vs. NN configuration'],
            # [['kinematic residual model (1x16 MLP sym)',
            #      'kinematic residual model (1x24 MLP sym)',
            #      'kinematic residual model (1x32 MLP)',
            #      'kinematic residual model (2x16 MLP)'],[0.072,0.109,0.101,0.09], 'kin+nn'],
            # [['black-box model (1x16 MLP sym)',
            #   'black-box model (1x24 MLP)',
            #   'black-box model (1x32 MLP)',
            #   'test',],[0.0866, 0.061, 0.099,0.0], 'nn'],
            ]

    for names,values, title in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = np.arange(len(names)) + 0.5
        ax.barh(pos[::-1], values, 0.4, color=['g', 'g', 'g', 'r', 'g', 'r', 'r', 'r', 'g', 'r', 'r'])
        plt.yticks(pos[::-1], names, horizontalalignment="left")
        ax.tick_params(axis="y", pad=210)
        ax.set_xlabel('solve time '+r'[s]')
        ax.grid('on', axis='x')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        # chartBox = ax.get_position()
        # ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.8, chartBox.height])
        # ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.9, 1.1), ncol=1, framealpha=1.0)
        ax.legend(handles=legend_elements, framealpha=1.0)
        fig.tight_layout()
        plt.show()

def accuracy_vs_NNsize():

    # data = [[[2.625, 2.285, 1.029, 0.758],'dyn+nn'], [[2.625, 2.285, 1.029, 0.758],'kin+nn']]
    data = [[[
        'dynamic model (M.H.)',
        'dynamic residual model (0x0 MLP sym)',
        'dynamic residual model (1x16 MLP)',
        'dynamic residual model (2x16 MLP)',
        'kinematic residual model (1x16 MLP)',
        'kinematic residual model (1x24 MLP sym)',
        'kinematic residual model (1x32 MLP)',
        'kinematic residual model (2x16 MLP)',
        'black-box model (1x16 MLP sym)',
        'black-box model (1x24 MLP)',
        'black-box model (1x32 MLP)',
    ], [2.625, 2.285, 1.029, 0.758, 0.106, 0.103, 0.101, 0.097, 0.097, 0.095, 0.093],
        ''],
        # 'Estimation performance vs. NN configuration'],
            # [['kinematic residual model (1x16 MLP sym)',
            #   'kinematic residual model (1x24 MLP sym)',
            #   'kinematic residual model (1x32 MLP)',
            #   'kinematic residual model (2x16 MLP)'], [0.106, 0.103, 0.101, 0.097], 'kin+nn'],
            # [['black-box model (1x16 MLP sym)',
            #   'black-box model (1x24 MLP)',
            #   'black-box model (1x32 MLP)',
            #   'test',], [0.097, 0.095, 0.093, 0.0], 'nn'],
            ]

    for names,values, title in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = np.arange(len(names)) + 0.5
        ax.barh(pos[::-1], values, 0.4,color=['g', 'g', 'g', 'r', 'g', 'r', 'r', 'r', 'g', 'r', 'r'])
        plt.yticks(pos[::-1], names, horizontalalignment="left")
        ax.tick_params(axis="y", pad=210)
        ax.set_xlabel('average mean squared error ' + r'[1]')
        ax.grid('on', axis='x')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        ax.legend(handles=legend_elements, framealpha=1.0)
        fig.tight_layout()
        plt.show()
    from matplotlib import gridspec
    fig = plt.figure(11, figsize=(8, 5))
    for names,values, title in data:
        gs = gridspec.GridSpec(2,1, height_ratios=[1,2])
        ax1 = plt.subplot(gs[0])
        pos = np.arange(len(names)) + 0.5
        print(pos[2::-1], values[:3])
        ax1.barh(pos[3::-1], values[:4], 0.4, color=['g', 'g', 'g', 'r', 'g', 'r', 'r', 'r', 'g', 'r', 'r'])
        plt.yticks(pos[3::-1], names[:4], horizontalalignment="left")
        ax1.tick_params(axis="y", pad=210)
        # ax1.set_xlabel('mean squared error ' + r'$[n/a]$')
        ax1.grid('on', axis='x')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        # ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        ax1.legend(handles=legend_elements, framealpha=1.0)

        ax2 = plt.subplot(gs[1])
        pos = np.arange(len(names)) + 0.5
        ax2.barh(pos[6::-1], values[4:], 0.4, color=['g', 'r', 'r', 'r', 'g', 'r', 'r'])
        plt.yticks(pos[6::-1], names[4:], horizontalalignment="left")
        ax2.tick_params(axis="y", pad=210)
        ax2.set_xlabel('average mean squared error ' + r'[1]')
        ax2.grid('on', axis='x')
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        # ax2.yaxis.set_minor_locator(AutoMinorLocator())
        # ax2.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        # ax2.legend(handles=legend_elements, framealpha=1.0)

        plt.tight_layout()


def delay_vs_stability():
    data = [[[
        'dynamic model (M.H.)',
        'dynamic model (M.H.) + 0.03 s',
        'dynamic model (M.H.) + 0.05 s',
        'dynamic model (M.H.) + 0.08 s',
    ],[0.055,0.082, 0.102, 0.128],
        ''],
               # 'Solve times vs. delay'],
    ]

    for names,values, title in data:
        fig, ax = plt.subplots(figsize=(8, 2.2))
        pos = np.arange(len(names))+0.5
        ax.barh(pos[::-1], values, 0.4, color=['g', 'g', 'r', 'r'])
        plt.yticks(pos[::-1], names, horizontalalignment="left")
        ax.tick_params(axis="y", pad=160)
        ax.set_xlabel('solve time '+r'[s]')
        ax.grid('on', axis='x')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        ax.legend(handles=legend_elements, framealpha=1.0)
        fig.tight_layout()
        plt.show()

def regularization_vs_stability():
    data = [[[
        'dynamic residual model (1x16 MLP), ' + r'$\lambda = 0.01$',
        'dynamic residual model (1x16 MLP), ' + r'$\lambda = 0.001$',
        'dynamic residual model (1x16 MLP), ' + r'$\lambda = 0.0001$',
    ],[0.091,0.095, 0.098],
        ''],
               # 'Solve times vs. regularization'],
    ]

    for names,values, title in data:
        fig, ax = plt.subplots(figsize=(8.5, 2))
        pos = np.arange(len(names))+0.5
        ax.barh(pos[::-1], values, 0.4, color=['g', 'r', 'r'])
        plt.yticks(pos[::-1], names, horizontalalignment="left")
        ax.tick_params(axis="y", pad=250)
        ax.set_xlabel('solve time '+r'[s]')
        ax.grid('on', axis='x')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(title)
        legend_elements = [Patch(facecolor='g',
                                 label='stable'),
                           Patch(facecolor='r',
                                 label='unstable')
                           ]
        fig.tight_layout()

        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height*0.75])
        ax.legend(title='', handles=legend_elements, framealpha=1.0, loc='upper center',bbox_to_anchor=(0.9, 1.6))
        plt.show()

def solvetime_vs_accuracy():
    plt.figure(20)
    plt.scatter([0.055,0.057,0.091,0.105], [2.625, 2.285, 1.029, 0.758])
    plt.xlabel('solve time' + r'[$s$]')
    plt.ylabel('mean squared error' + r'[-]')
    # plt.axis('equal')
    plt.xlim([0,0.11])
    plt.ylim([0,2.7])
    plt.show()

if __name__ == '__main__':
    # main()
    # accuracy_vs_NNsize()
    # delay_vs_stability()
    regularization_vs_stability()
    # solvetime_vs_accuracy()