#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:00:21 2019

@author: mvb
original code from MATLAB written by Marc Heim
"""
import numpy as np

def marc_vehiclemodel (V, U):
    #Parameters from optimizations (Michi's)
    # res = [2.4869, 7.6091 ,2.1568] #gradient descent
    res = [ 5.6698 ,18.1382 , 0.1669] #least squares
    D1 = res[0]
    D2 = res[1]
    Ic = res[2]


    #Optimal parameters (from Marc)
    # D1 = 9.4
    # D2 = 10.4
    # Ic = 0.3
    #
    # B1 = 15
    # C1 = 1.1
    # B2 = 5.2
    # C2 = 1.4

    #New optimal parameters (from Marc)
    B1 = 9;
    C1 = 1;
    B2 = 5.2;
    C2 = 1.1;
    # D2 = 10;
    # D1 = 10;

    param = [B1,C1,D1,B2,C2,D2,Ic]
    [accX,accY,accRot] = pymodelDx(V, U, param)   #Marc's Matlab function translated to python

    return accX,accY,accRot


def pymodelDx(V, U, param):
    global B1, B2, C1, C2, D1, D2, reg

    VELX, VELY, VELROTZ = V
    BETA, AB, TV = U

#    %param = [B1,C1,D1,B2,C2,D2,Ic];
    B1 = param[0]
    C1 = param[1]
    D1 = param[2]
    B2 = param[3]
    C2 = param[4]
    D2 = param[5]
    Ic = param[6]

    reg = 0.2 #default: 0.5

    l = 1.19
    l1 = 0.73
    l2 = l-l1
    f1n = l2/l
    f2n = l1/l
    w = 1.08
    # w = 0.

    vel1 = np.matmul(rotmat(BETA),np.array([[VELX],[VELY+l1*VELROTZ]]))
    f1y = simplefaccy(vel1[1],vel1[0])

    F1 = np.matmul(rotmat(-BETA),np.array([[0],[f1y[0]]]))*f1n
    F1x = F1[0]
    F1y = F1[1]
    # print('F1y',F1y*l1)
    F2x = AB
    F2y1 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB+TV/2.)/f2n)*f2n/2.
    #F2y1 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB) / f2n) * f2n / 2.
    F2y2 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB-TV/2.)/f2n)*f2n/2.
    #F2y2 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB) / f2n) * f2n / 2.
    F2y = simpleaccy(VELY-l2*VELROTZ,VELX,AB/f2n)*f2n
    # print('F2y',F2y*l2)
    TVTrq = TV*w
    # print('tv',TVTrq)

    ACCROTZ = (TVTrq + F1y * l1 - F2y * l2) / Ic
    ACCX = F1x+F2x+VELROTZ*VELY
    ACCY = F1y+F2y1+F2y2-VELROTZ*VELX
    
    return ACCX[0],ACCY[0],ACCROTZ[0]

def rotmat(beta):
    return np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta), np.cos(beta)]])

def magic(s, B, C, D):
    return D * np.sin(C * np.arctan(B * s))


def capfactor(taccx):
    return (1 - satfun((taccx / D2) ** 2)) ** (1 / 2)


def simpleslip(VELY, VELX, taccx):
    return -(1 / capfactor(taccx)) * VELY / (VELX + reg)


def simplediraccy(VELY, VELX, taccx):
    return magic(simpleslip(VELY, VELX, taccx), B2, C2, D2)


def simpleaccy(VELY, VELX, taccx):
    return capfactor(taccx) * simplediraccy(VELY, VELX, taccx)


def simplefaccy(VELY, VELX):
    return magic(-VELY / (VELX + reg), B1, C1, D1)


def satfun(x):
    l = 0.8;
    r = 1-l;
    if isinstance(x, float):
        if x<l:
            y=x;
        elif x<1+r:
            d = (1+r-x)/r;
            y = 1-1/4*r*d**2;
        else:
            y = 1;
        y=0.95*y;
    else:
        print('ERROR: x in satfun(x) is not float!')
    return y
