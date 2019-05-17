#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:00:21 2019

@author: mvb
original code from MATLAB written by Marc Heim
"""
import numpy as np

def marc_vehiclemodel (X,W):

    #optimal parameters
    B1 = 15
    C1 = 1.1
    # D1 = 9.4
    B2 = 5.2
    C2 = 1.4
    # D2 = 10.4
    # Ic = 0.3

    # param = [B1,C1,D1,B2,C2,D2,Ic]
    param = [B1,C1,B2,C2]
    accelerations = pymodelDx(X, W, param)   #Marc's Matlab function translated to python

    return accelerations


# def Jacobian_marc_vehiclemodel (X,W):
#
#     #optimal parameters
#     B1 = 15
#     C1 = 1.1
#     # D1 = 9.4
#     B2 = 5.2
#     C2 = 1.4
#     # D2 = 10.4
#     # Ic = 0.3
#
#     # param = [B1,C1,D1,B2,C2,D2,Ic]
#     param = [B1,C1,B2,C2]
#     J = jacobian_of_pymodelDx(X, W, param)   #Marc's Matlab function translated to python
#
#     return J


def pymodelDx(X, W, param):
    global B1, B2, C1, C2, D1, D2, reg
    if len(X.shape)>1:
        VELX = X[:, 0]
        VELY = X[:, 1]
        VELROTZ = X[:, 2]
        BETA = X[:, 3]
        AB = X[:, 4]
        TV = X[:, 5]
    else:
        VELX, VELY, VELROTZ, BETA, AB, TV = X

    D1, D2, Ic = W

    #    %param = [B1,C1,D1,B2,C2,D2,Ic];
    B1 = param[0]
    C1 = param[1]
    # D1 = param[2]
    B2 = param[2]
    C2 = param[3]
    # D2 = param[5]
    # Ic = param[6]

    reg = 0.2  # default: 0.5

    l = 1.19
    l1 = 0.73
    l2 = l - l1
    f1n = l2 / l
    f2n = l1 / l
    w = 1.08

    velx = np.sum(np.multiply(rotmat(BETA)[0], np.array([VELX, VELY + l1 * VELROTZ])),axis=0)
    vely = np.sum(np.multiply(rotmat(BETA)[1], np.array([VELX, VELY + l1 * VELROTZ])), axis=0)
    f1y = simplefaccy(vely, velx)
    # vel1 = np.matmul(rotmat(BETA), np.array([[VELX], [VELY + l1 * VELROTZ]]))
    # f1y = simplefaccy(vel1[1], vel1[0])

    F1x = np.sum(np.multiply(rotmat(-BETA)[0], np.array([np.zeros(len(f1y)), f1y])), axis=0)
    F1y = np.sum(np.multiply(rotmat(-BETA)[1], np.array([np.zeros(len(f1y)), f1y])), axis=0)
    # F1 = np.concatenate((F1_x,F1_y)) * f1n
    # F1 = np.matmul(rotmat(-BETA), np.array([[0], [f1y[0]]])) * f1n
    # F1x = F1[0]
    # F1y = F1[1]

    F2x = AB
    F2y1 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB + TV / 2.) / f2n) * f2n / 2.
    F2y2 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB - TV / 2.) / f2n) * f2n / 2.
    F2y = simpleaccy(VELY - l2 * VELROTZ, VELX, AB / f2n) * f2n
    TVTrq = TV * w

    ACCROTZ = (TVTrq + D1 * F1y * l1 - D2 * F2y * l2) / Ic
    ACCX = D1 * F1x + F2x + VELROTZ * VELY
    ACCY = D1 * F1y + D2 * (F2y1 + F2y2) - VELROTZ * VELX

    Jacobian = np.array([[F1x, np.zeros(len(F1x)), np.zeros(len(F1x))],
                         [F1y, F2y1 + F2y2, np.zeros(len(F1y))],
                         [F1y * l1 / Ic, F2y * l2 / Ic,
                          -(TVTrq + D1 * F1y * l1 - D2 * F2y * l2) / np.square(Ic)]]).transpose()

    return np.array([ACCX, ACCY, ACCROTZ]).transpose(), np.array(list(Jacobian))


# def jacobian_of_pymodelDx(X, W, param):
#     global B1, B2, C1, C2, D1, D2, reg
#
#     VELX = X[:, 0]
#     VELY = X[:, 1]
#     VELROTZ = X[:, 2]
#     BETA = X[:, 3]
#     AB = X[:, 4]
#     TV = X[:, 5]
#
#     D1, D2, Ic = W
#
#     #    %param = [B1,C1,D1,B2,C2,D2,Ic];
#     B1 = param[0]
#     C1 = param[1]
#     # D1 = param[2]
#     B2 = param[2]
#     C2 = param[3]
#     # D2 = param[5]
#     # Ic = param[6]
#
#     reg = 0.2 #default: 0.5
#
#     l = 1.19
#     l1 = 0.73
#     l2 = l-l1
#     f1n = l2/l
#     f2n = l1/l
#     w = 1.08
#
#     velx = np.sum(np.multiply(rotmat(BETA)[0], np.array([VELX, VELY + l1 * VELROTZ])), axis=0)
#     vely = np.sum(np.multiply(rotmat(BETA)[1], np.array([VELX, VELY + l1 * VELROTZ])), axis=0)
#     f1y = simplefaccy(vely, velx)
#     # vel1 = np.matmul(rotmat(BETA),np.array([[VELX],[VELY+l1*VELROTZ]]))
#     # f1y = simplefaccy(vel1[1],vel1[0])
#
#     F1x = np.sum(np.multiply(rotmat(-BETA)[0], np.array([np.zeros(len(f1y)), f1y])), axis=0)
#     F1y = np.sum(np.multiply(rotmat(-BETA)[1], np.array([np.zeros(len(f1y)), f1y])), axis=0)
#     # F1 = np.matmul(rotmat(-BETA),np.array([[0],[f1y[0]]]))*f1n
#     # F1x = F1[0]
#     # F1y = F1[1]
#     F2x = AB
#     F2y1 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB+TV/2.)/f2n)*f2n/2.
#     F2y2 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB-TV/2.)/f2n)*f2n/2.
#     F2y = simpleaccy(VELY-l2*VELROTZ,VELX,AB/f2n)*f2n
#     TVTrq = TV*w
#
#     # ACCROTZ = (TVTrq + F1y * l1 - F2y * l2) / Ic
#     # ACCX = F1x+F2x+VELROTZ*VELY
#     # ACCY = F1y+F2y1+F2y2-VELROTZ*VELX
#
#     Jacobian = np.array([[F1x, np.zeros(len(F1x)), np.zeros(len(F1x))],
#                 [F1y, F2y1 + F2y2, np.zeros(len(F1y))],
#                 [F1y * l1 / Ic, F2y * l2 / Ic, -(TVTrq + D1 * F1y * l1 - D2 * F2y * l2) / np.square(Ic)]]).transpose()
#
#     return np.array(list(Jacobian))


def rotmat(beta):
    return np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta), np.cos(beta)]])


def magic(s, B, C):
    return np.sin(C * np.arctan(B * s))


def capfactor(taccx):
    return (1 - satfun_approx((taccx / D2) ** 2)) ** (1 / 2)


def simpleslip(VELY, VELX, taccx):
    return -(1 / capfactor(taccx)) * VELY / (VELX + reg)


def simplediraccy(VELY, VELX, taccx):
    return magic(simpleslip(VELY, VELX, taccx), B2, C2)


def simpleaccy(VELY, VELX, taccx):
    return capfactor(taccx) * simplediraccy(VELY, VELX, taccx)


def simplefaccy(VELY, VELX):
    return magic(-VELY / (VELX + reg), B1, C1)


def satfun_approx(x):
    return (1.03616764e+00 * x + -1.03619930e+00) * (-2.91844486e-01 * np.arctan(7.27342214e+00 * (x + -1.04920374e+00) + 3.59055456e-01) + 4.58406776e-01) + (
                -3.10005810e-02 * np.arctan(1.70845190e-02 * (x + -9.54399532e+01) + 6.32433105e-01) + -1.97634176e+01) * (-4.60294363e-02)
