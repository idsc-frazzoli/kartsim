#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 09.05.19 10:11

@author: mvb
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interpolation_w_mean(x, y, xBegin, xStop, timeStep):
    if isinstance(y, pd.Series):
        interp = interp1d(x, y)

        if xBegin % timeStep != 0:
            xBegin = (timeStep - xBegin % timeStep) + xBegin
        else:
            pass

        if xStop % timeStep != 0:
            xStop = xStop - xStop % timeStep
        else:
            pass

        xInterp = np.linspace(xBegin, xStop, int((xStop - xBegin) / timeStep)*10)

        yInterp = interp(xInterp)

        return xInterp, yInterp

    elif isinstance(y, list):
        interp = interp1d(x, y)

        if xBegin % timeStep != 0:
            xBegin = np.around((timeStep - xBegin % timeStep) + xBegin, 2)
        else:
            pass

        if xStop % timeStep != 0:
            xStop = np.around(xStop - xStop % timeStep, 2)
        else:
            pass

        xInterp = np.linspace(xBegin, xStop, int(np.around((xStop - xBegin) / timeStep)) + 1)

        x_eval = np.linspace(xBegin, xStop, int(np.around((xStop - xBegin) / timeStep)) * 10 + 1)
        x_eval = np.round(x_eval, 2)
        y_eval = interp(x_eval)

        yInterp = np.array([])
        for i, x in enumerate(xInterp):
            if i == 0:
                yInterp = np.array([np.mean(y_eval[:i * 10 + 5])])
            elif i == len(xInterp)-1:
                # yInterp = np.hstack((yInterp, np.mean(y_eval[i * 100 - 50:i * 100 + 51])))
                yInterp = np.hstack((yInterp, np.mean(y_eval[i * 10 - 5:])))
            else:
                yInterp = np.hstack((yInterp, np.mean(y_eval[i * 10 - 5:i * 10 + 5])))

        return xInterp, yInterp

def interpolation(x, y, xBegin, xStop, timeStep):
    if isinstance(y, pd.Series):
        interp = interp1d(x, y)

        if xBegin % timeStep != 0:
            xBegin = (timeStep - xBegin % timeStep) + xBegin
        else:
            pass

        if xStop % timeStep != 0:
            xStop = xStop - xStop % timeStep
        else:
            pass

        xInterp = np.linspace(xBegin, xStop, int((xStop - xBegin) / timeStep))
        yInterp = interp(xInterp)

        return xInterp, yInterp

    elif isinstance(y, list):
        interp = interp1d(x, y)

        if xBegin % timeStep != 0:
            xBegin = np.around((timeStep - xBegin % timeStep) + xBegin, 2)
        else:
            pass

        if xStop % timeStep != 0:
            xStop = np.around(xStop - xStop % timeStep, 2)
        else:
            pass

        xInterp = np.linspace(xBegin, xStop, int(np.around((xStop - xBegin) / timeStep)) + 1)
        xInterp = np.round(xInterp, 2)

        yInterp = interp(xInterp)

        return xInterp, yInterp


def derivative_X_dX(name, x, y):
    if name == 'pose vtheta':
        dydx = np.diff(y) / np.diff(x)
        lim = 4
        for i in range(len(dydx) - 1):
            if dydx[i] > lim:
                dydx[i] = (y[i + 1] - y[i] - 2 * np.pi) / (x[i + 1] - x[i])
            elif dydx[i] < -lim:
                dydx[i] = (y[i + 1] - y[i] + 2 * np.pi) / (x[i + 1] - x[i])
                # k = 1
                # while dydx[i + k] > lim or dydx[i + k] < -lim:
                #     k += 1
                # for l in range(k):
                #     dydx[i + l] = (dydx[i + l - 1] + dydx[i + k]) / 2.0

    else:
        dydx = np.diff(y) / np.diff(x)
    return x[:-1], dydx