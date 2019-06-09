#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:00:21 2019

@author: mvb
original code from MATLAB written by Marc Heim
"""
import numpy as np
import tensorflow as tf

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
    [accX,accY,accRot] = pymodelDx(X, W, param)   #Marc's Matlab function translated to python

    return accX,accY,accRot


def pymodelDx(X, W, param):
    global B1, B2, C1, C2, D1, D2, reg
    VELX = X[0,0]
    VELY = X[0,1]
    VELROTZ = X[0,2]
    BETA = X[0,3]
    AB = X[0,4]
    TV = X[0,5]

    D1 = W[0]
    D2 = W[1]
    w = W[2]
    Ic = W[3]

#    %param = [B1,C1,D1,B2,C2,D2,Ic];
    B1 = param[0]
    C1 = param[1]
    # D1 = param[2]
    B2 = param[2]
    C2 = param[3]
    # D2 = param[5]
    # Ic = param[6]



    reg = 0.2 #default: 0.5

    l = 1.19
    l1 = 0.73
    l2 = l-l1
    f1n = l2/l
    f2n = l1/l
    # w = 1.08

    vel1 = tf.matmul(rotmat_tf(BETA),tf.stack([[VELX],[VELY+l1*VELROTZ]]))
    f1y = simplefaccy(vel1[1],vel1[0])

    F1 = tf.matmul(rotmat_tf(-BETA),tf.stack([[0],[f1y[0]]]))*f1n
    F1x = F1[0]
    F1y = F1[1]
    F2x = AB
    F2y1 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB+TV/2.)/f2n)*f2n/2.
    F2y2 = simpleaccy(VELY-l2*VELROTZ,VELX,(AB-TV/2.)/f2n)*f2n/2.
    F2y = simpleaccy(VELY-l2*VELROTZ,VELX,AB/f2n)*f2n
    TVTrq = TV*w

    ACCROTZ = (TVTrq + F1y * l1 - F2y * l2) / Ic
    ACCX = F1x+F2x+VELROTZ*VELY
    ACCY = F1y+F2y1+F2y2-VELROTZ*VELX
    
    return ACCX[0], ACCY[0], ACCROTZ[0]

# def _rotmat(beta):
#     return np.array([[tf.cos(beta),tf.sin(beta)],[-tf.sin(beta), tf.cos(beta)]])
def rotmat_tf(beta):
    R1 = tf.stack([tf.cos(beta),tf.sin(beta)],axis=0)
    R2 = tf.stack([-tf.sin(beta), tf.cos(beta)],axis=0)
    return tf.stack([R1, R2],axis=0)

def magic(s, B, C, D):
    return D * tf.sin(C * tf.atan(B * s))


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

    def ft0(x,l,r): return tf.cond(tf.less(x,l), lambda: ft1(x), lambda: ff1(x,r))

    def ff0(): return 999.0

    def ft1(x): return x

    def ff1(x,r): return tf.cond(tf.less(x,1+r), lambda: ft2(x,r), lambda: ff2())

    def ft2(x,r):
        d = (1+r-x)/r
        return 1-1/4*r*d**2

    def ff2(): return 1.0

    y = tf.cond(tf.equal(isinstance(x,tf.Tensor),True), lambda: ft0(x,l,r), lambda: ff0())

    if y == 999.0:
        print('ERROR: x in _satfun(x) is not float!')
        return None
    else:
        y = 0.95 * y
        return y
    # if isinstance(x, tf.Tensor):
    #     if x<l:
    #         y=x;
    #     elif x<1+r:
    #         d = (1+r-x)/r;
    #         y = 1-1/4*r*d**2;
    #     else:
    #         y = 1;
    #     y=0.95*y;
    # else:
    #     print('ERROR: x in _satfun(x) is not float!')

