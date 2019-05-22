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
    # B1 = 15
    # C1 = 1.1
    # D1 = 9.4
    # B2 = 5.2
    # C2 = 1.4
    # D2 = 10.4
    # Ic = 0.3

    # New optimal parameters (from Marc)
    B1 = 9;
    C1 = 1;
    B2 = 5.2;
    C2 = 1.1;
    # D2 = 10;
    # D1 = 10;

    # param = [B1,C1,D1,B2,C2,D2,Ic]
    param = [B1,C1,B2,C2]
    accelerations,J = pymodelDx(X, W, param)   #Marc's Matlab function translated to python

    return accelerations,J


def Jacobian_marc_vehiclemodel (X,W):

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
    J = jacobian_of_pymodelDx(X, W, param)   #Marc's Matlab function translated to python

    return J


def pymodelDx(X, W, param):
    global B1, B2, C1, C2, D1, D2, reg

    VELX = X[:, 0]
    VELY = X[:, 1]
    VELROTZ = X[:, 2]
    BETA = X[:, 3]
    AB = X[:, 4]
    TV = X[:, 5]

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

    # print('f1y_Jc', f1y[:3])

    # vel1 = np.matmul(rotmat(BETA), np.array([[VELX], [VELY + l1 * VELROTZ]]))
    # f1y = simplefaccy(vel1[1], vel1[0])

    F1x = np.sum(np.multiply(rotmat(-BETA)[0], np.array([np.zeros(len(f1y)), f1y])), axis=0)*f1n
    F1y = np.sum(np.multiply(rotmat(-BETA)[1], np.array([np.zeros(len(f1y)), f1y])), axis=0)*f1n


    # F1 = np.concatenate((F1_x,F1_y)) * f1n
    # F1 = np.matmul(rotmat(-BETA), np.array([[0], [f1y[0]]])) * f1n
    # F1x = F1[0]
    # F1y = F1[1]

    F2x = AB

    F2y1 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB + TV / 2.) / f2n) * f2n / 2.
    F2y2 = simpleaccy(VELY - l2 * VELROTZ, VELX, (AB - TV / 2.) / f2n) * f2n / 2.
    F2y = simpleaccy(VELY - l2 * VELROTZ, VELX, AB / f2n) * f2n
    TVTrq = TV * w

    ACCROTZ = (TVTrq + F1y * l1 - F2y * l2) / Ic
    ACCX = F1x + F2x + VELROTZ * VELY
    ACCY = F1y + (F2y1 + F2y2) - VELROTZ * VELX

    F2y_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, AB / f2n) * f2n
    F2y1_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, (AB + TV / 2.) / f2n) * f2n / 2.
    F2y2_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, (AB - TV / 2.) / f2n) * f2n / 2.

    Jacobian = np.array([[F1x/D1, np.zeros(len(F1x)), np.zeros(len(F1x))],
                         [F1y/D1, F2y1_dD2 + F2y2_dD2, np.zeros(len(F1y))],
                         [F1y/D1 * l1 / Ic, F2y_dD2 * l2 / Ic,
                          -(TVTrq + F1y * l1 - F2y * l2) / np.square(Ic)]]).transpose()

    return np.array([ACCX, ACCY, ACCROTZ]).transpose(), np.array(list(Jacobian))


def jacobian_of_pymodelDx(X, W, param):
    global B1, B2, C1, C2, D1, D2, reg

    VELX = X[:, 0]
    VELY = X[:, 1]
    VELROTZ = X[:, 2]
    BETA = X[:, 3]
    AB = X[:, 4]
    TV = X[:, 5]

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

    velx = np.sum(np.multiply(rotmat(BETA)[0], np.array([VELX, VELY + l1 * VELROTZ])), axis=0)
    vely = np.sum(np.multiply(rotmat(BETA)[1], np.array([VELX, VELY + l1 * VELROTZ])), axis=0)
    f1y = simplefaccy(vely, velx)
    print('f1y_Jc',f1y)

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

    F2y_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, AB / f2n) * f2n
    F2y1_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, (AB + TV / 2.) / f2n) * f2n / 2.
    F2y2_dD2 = simpleaccy_dD2(VELY - l2 * VELROTZ, VELX, (AB - TV / 2.) / f2n) * f2n / 2.

    Jacobian = np.array([[F1x, np.zeros(len(F1x)), np.zeros(len(F1x))],
                [F1y, F2y1_dD2 + F2y2_dD2, np.zeros(len(F1y))],
                [F1y * l1 / Ic, F2y_dD2 * l2 / Ic, -(TVTrq + D1 * F1y * l1 - D2 * F2y * l2) / np.square(Ic)]]).transpose()

    return np.array(list(Jacobian))


def rotmat(beta):
    return np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta), np.cos(beta)]])


def magic(s, B, C, D):
    return D * np.sin(C * np.arctan(B * s))


def magic_derivative(B, C, D, VELY, VELX, taccx):
    s = simpleslip(VELY, VELX, taccx)
    return D * np.cos(C * np.arctan(B * s)) * C * 1./(1 + (B * s)**2) * B * simpleslip_dD2(VELY, VELX, taccx)


# def s_dD2(B):
#     return 0.5 * B * VELY / (VELX + reg) * (1-satfun_approx(taccx)) ** (-3/2) * satfun_approx_derivative(taccx) * -2 * taccx * D2


def capfactor(taccx):
    return (1 - satfun_approx((taccx / D2) ** 2)) ** (1 / 2.)


def capfactor_dD2(taccx):
    # print('satfun',0.5 * 1./(1-satfun_approx(taccx)[0])**(1/2.))
    # print(satfun_approx_derivative(taccx)[0])
    # print(taccx[0])
    # print(taccx[0]**2)
    return 0.5 * 1./(1-satfun_approx(taccx))**(1/2.) * satfun_approx_derivative(taccx) * -2 * taccx**2 / D2


def simpleslip(VELY, VELX, taccx):
    return -(1 / capfactor(taccx)) * VELY / (VELX + reg)


def simpleslip_dD2(VELY, VELX, taccx):
    return 0.5 * VELY / (VELX + reg) * (1 - satfun_approx(taccx)) ** (-3 / 2.) * satfun_approx_derivative(
        taccx) * -2 * taccx / D2


def simplediraccy(VELY, VELX, taccx):
    return magic(simpleslip(VELY, VELX, taccx), B2, C2, D2)


def simpleaccy(VELY, VELX, taccx):
    return capfactor(taccx) * simplediraccy(VELY, VELX, taccx)


def simpleaccy_dD2(VELY, VELX, taccx):
    linear_part = simpleaccy(VELY, VELX, taccx) / D2
    cap_part = capfactor_dD2(taccx) * simplediraccy(VELY, VELX, taccx)
    magic_part = capfactor(taccx) * magic_derivative(B2, C2, D2, VELY, VELX, taccx)

    # print('linear',linear_part[0])
    # print('cap',cap_part[0])
    # print(' ',capfactor_dD2(taccx)[0])
    # print(' ',simplediraccy(VELY, VELX, taccx)[0])
    # print('magic',magic_part[0])
    # # print(capfactor(taccx)[0])
    # # print(magic_derivative(B2, C2, D2, VELY, VELX, taccx)[0])
    # print('TOTAL', linear_part[0]+cap_part[0]+magic_part[0], '\n')

    return linear_part + cap_part + magic_part
    # return simpleaccy(VELY, VELX, taccx) / D2 + capfactor_dD2(taccx) * simplediraccy(VELY, VELX, taccx) + capfactor(
    #     taccx) * magic_derivative(B2, C2, D2, VELY, VELX, taccx)


def simplefaccy(VELY, VELX):
    return magic(-VELY / (VELX + reg), B1, C1, D1)


def satfun_approx(x):
    a = 1.03616764e+00
    b = -1.03619930e+00
    c = -2.91844486e-01
    d = 7.27342214e+00
    e = -1.04920374e+00
    f = 3.59055456e-01
    g = 4.58406776e-01
    h = -3.10005810e-02
    i = 1.70845190e-02
    j = -9.54399532e+01
    k = 6.32433105e-01
    l = -1.97634176e+01
    m = -4.60294363e-02
    return (a * x + b) * (c * np.arctan(d * (x + e) + f) + g) + (h * np.arctan(i * (x + j) + k) + l) * (m)

def satfun_approx_derivative(x):
    a = 1.03616764e+00
    b = -1.03619930e+00
    c = -2.91844486e-01
    d = 7.27342214e+00
    e = -1.04920374e+00
    f = 3.59055456e-01
    g = 4.58406776e-01
    h = -3.10005810e-02
    i = 1.70845190e-02
    j = -9.54399532e+01
    k = 6.32433105e-01
    l = -1.97634176e+01
    m = -4.60294363e-02

    return a * (c * np.arctan(d * (x + e) + f) + g) + (a * x + b) * c * d * (1 + (d * (x + e) + f) ** 2) ** (
        -1) + h * m * i * (1 + (i * (x + j) + k) ** 2) ** (-1)
