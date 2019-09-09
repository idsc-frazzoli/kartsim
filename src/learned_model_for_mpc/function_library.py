#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 01.09.19 00:20

@author: mvb
"""
import time

import numpy as np
from itertools import combinations_with_replacement
from pandas import DataFrame


def main():
    # arr = np.arange(1,2040001).reshape(6,-1)
    arr = np.arange(1, 19).reshape(-1, 6)
    arr = DataFrame(arr, columns=['VX', 'VY', 'VTHETA', 'BETA', 'AB', 'TV'])
    arr_feature_names = []
    for name in arr.columns:
        arr_feature_names.append(name)
    arr = arr.values.T
    features = arr
    feature_names = arr_feature_names.copy()
    print(features.shape)
    print(features)
    print(feature_names)

    # res, new_feature_names = apply_powers(arr, arr_feature_names)
    # features = np.vstack((features, res))
    # feature_names += new_feature_names
    # print(features.shape)
    # for i in range(len(features)):
    #     print(features[i], feature_names[i])

    res, new_feature_names = apply_exponentials(arr, arr_feature_names)
    features = np.vstack((features, res))
    feature_names += new_feature_names
    print(features.shape)

    res, new_feature_names = apply_trigo(arr, arr_feature_names)
    features = np.vstack((features, res))
    feature_names += new_feature_names
    print(features.shape)
    # print(features)
    print(new_feature_names)

    # res, new_feature_names = add_custom_features(arr, arr_feature_names)
    # features = np.vstack((features, res))
    # feature_names += new_feature_names
    # print(features.shape)
    # # for i in range(len(features)):
    # #     print(features[i], feature_names[i])

    features, feature_names = get_polynomials(features, feature_names, degree=3)
    print(features.shape)
    for i in range(len(features)):
        print(features[i], feature_names[i])


def get_new_features(orig_features):
    if isinstance(orig_features, DataFrame):
        orig_feature_names = []
        for name in orig_features.columns:
            if name == 'vehicle vx [m*s^-1]':
                name = 'VX'
            elif name == 'vehicle vy [m*s^-1]':
                name = 'VY'
            elif name == 'pose vtheta [rad*s^-1]':
                name = 'VTHETA'
            elif name == 'turning angle [n.a]':
                name = 'BETA'
            elif name == 'acceleration rear axle [m*s^-2]':
                name = 'AB'
            elif name == 'acceleration torque vectoring [rad*s^-2]':
                name = 'TV'
            orig_feature_names.append(name)
        orig_features_copy = orig_features.copy()
        orig_features_copy = orig_features_copy.values.T
    elif isinstance(orig_features, np.ndarray):
        orig_features_copy = np.copy(orig_features)
        orig_features_copy = orig_features_copy.T
    else:
        raise ValueError('Unsuported data format.')
    orig_features_copy[orig_features_copy == 0] = 0.0000001
    features = orig_features_copy.copy()
    feature_names = orig_feature_names.copy()
    print(features.shape)

    # res, new_feature_names = apply_powers(orig_features_copy, orig_feature_names)
    # features = np.vstack((features, res))
    # feature_names += new_feature_names
    # print(features.shape)

    res, new_feature_names = apply_exponentials(orig_features_copy, orig_feature_names)
    features = np.vstack((features, res))
    feature_names += new_feature_names
    print(features.shape)

    res, new_feature_names = apply_trigo(orig_features_copy, orig_feature_names)
    features = np.vstack((features, res))
    feature_names += new_feature_names
    print(features.shape)

    # res, new_feature_names = add_custom_features(orig_features_copy, orig_feature_names)
    # features = np.vstack((features, res))
    # feature_names += new_feature_names
    # print(features.shape)

    features, feature_names = get_polynomials(features, feature_names, degree=2)
    print(features.shape)

    if isinstance(orig_features, DataFrame):
        features = DataFrame(features.T, columns=feature_names)
    elif isinstance(orig_features, np.ndarray):
        features = features.T
    return features


def combinations(features, degree=2):
    res = np.array(list(combinations_with_replacement(features, degree)))
    return res


def get_polynomials(features, feature_names, degree=2):
    new_features = features
    for repeat in range(degree - 1):
        combis = combinations(range(features.shape[0]), degree=2 + repeat)

        # sort out trivial cases (e.g. vx * 1/vx)
        remove_combi = []
        for i, combi in enumerate(combis):
            if ',1/2.0' in feature_names[combi[0]] and feature_names[combi[0]] == feature_names[combi[1]]:
                remove_combi.append(i)
            # elif combi[0] < 6 and '^(-1)' in feature_names[combi[1]] and feature_names[combi[0]] in feature_names[
            #     combi[1]]:
            #     remove_combi.append(i)
            elif 'sin(' in feature_names[combi[0]] and (
                    'cos(' in feature_names[combi[1]] or 'tan(' in feature_names[combi[1]]) and \
                    feature_names[combi[0]][3:] in feature_names[combi[1]]:
                remove_combi.append(i)
            elif 'cos(' in feature_names[combi[0]] and 'tan(' in feature_names[combi[1]] and \
                    feature_names[combi[0]][3:] in feature_names[combi[1]]:
                remove_combi.append(i)
            elif '(2,' in feature_names[combi[0]] and '(1/2.0,' in feature_names[combi[1]] and \
                    feature_names[combi[0]][11:] == feature_names[combi[1]][15:]:
                remove_combi.append(i)

            if 2 + repeat == 3:
                for pair in [[0,2],[1,2],[1,0],[2,0],[2,1],]:
                    if ',1/2.0' in feature_names[combi[pair[0]]] and feature_names[combi[pair[0]]] == feature_names[combi[pair[1]]]:
                        remove_combi.append(i)
                    elif combi[pair[0]] < 6 and '^(-1)' in feature_names[combi[pair[1]]] and feature_names[combi[pair[0]]] in feature_names[
                        combi[pair[1]]]:
                        remove_combi.append(i)
                    elif 'sin(' in feature_names[combi[pair[0]]] and \
                            ('cos(' in feature_names[combi[pair[1]]] or 'tan(' in feature_names[combi[pair[1]]]) and \
                            feature_names[combi[pair[0]]][3:] in feature_names[combi[pair[1]]]:
                        remove_combi.append(i)
                    elif 'cos(' in feature_names[combi[pair[0]]] and 'tan(' in feature_names[combi[pair[1]]] and \
                            feature_names[combi[pair[0]]][3:] in feature_names[combi[pair[1]]]:
                        remove_combi.append(i)
                    elif '(2,' in feature_names[combi[pair[0]]] and '(1/2.0,' in feature_names[combi[pair[1]]] and \
                            feature_names[combi[pair[0]]][11:] == feature_names[combi[pair[1]]][15:]:
                        remove_combi.append(i)
                        # print(pair[0], pair[1], feature_names[combi[pair[0]]], feature_names[combi[pair[1]]])

        # print('-', len(remove_combi))
        combis = np.delete(combis, remove_combi, axis=0)

        res = features[combis[:, 0]]
        for n in range(2 + repeat - 1):
            res = np.multiply(res, features[combis[:, n + 1]])
        new_features = np.vstack((new_features, res))
        new_feature_names = []
        if 2 + repeat == 2:
            for combi in combis:
                new_feature_names.append(f'({feature_names[combi[0]]}) * ({feature_names[combi[1]]})')
        if 2 + repeat == 3:
            for combi in combis:
                new_feature_names.append(
                    f'({feature_names[combi[0]]}) * ({feature_names[combi[1]]}) * ({feature_names[combi[2]]})')
        feature_names += new_feature_names
    return new_features, feature_names


def apply_powers(features, feature_names):
    new_features = np.power(np.abs(features), 1 / 2.0)
    # new_features = np.vstack((new_features, np.power(features, 1 / 3.0)))
    new_features = np.vstack((new_features, np.power(features, -1.0)))
    new_feature_names = []
    for calculation in ('1/2.0', '-1.0'):
        for name in feature_names:
            new_feature_names.append(f'np.power({name},{calculation})')
    return new_features, new_feature_names


def apply_exponentials(features, feature_names):
    new_features = np.power(2, features)
    new_features = np.vstack((new_features, np.power(1 / 2.0, features)))
    # new_features = np.vstack((new_features, np.log(np.abs(features))))
    new_feature_names = []
    for calculation in ('2', '1/2.0'):
        for name in feature_names:
            new_feature_names.append(f'np.power({calculation},{name})')
    remove_list = []
    for i, name in enumerate(new_feature_names):
        if any(tag in name for tag in ('VY', 'VTHETA', 'BETA', 'TV', 'turning angle', 'torque vectoring')):
            remove_list.append(i)
    # print('-', len(remove_list))
    new_features = np.delete(new_features, remove_list, axis=0)
    new_feature_names = list(np.delete(new_feature_names, remove_list, axis=0))
    return new_features, new_feature_names


def apply_trigo(features, feature_names):
    new_features = np.sin(
        np.multiply(np.divide(features, np.tile(np.array([[10.0,4.0,2.8,0.44,6.7,2.1]]).T, features.shape[1])), np.pi / 2.0))
    new_features = np.vstack((new_features, np.cos(
        np.multiply(np.divide(features, np.tile(np.array([[10.0,4.0,2.8,0.44,6.7,2.1]]).T, features.shape[1])), np.pi / 2.0))))
    new_features = np.vstack((new_features, np.tan(
        np.multiply(np.divide(features, np.tile(np.array([[10.0,4.0,2.8,0.44,6.7,2.1]]).T, features.shape[1])), np.pi / 3.0))))
    new_feature_names = []
    for calculation in ('sin', 'cos'):
        for name, divisor in zip(feature_names, [10.0,4.0,2.8,0.44,6.7,2.1]):
            new_feature_names.append(f'{calculation}({name}/{divisor}*pi/2.0)')
    for name, divisor in zip(feature_names, [10.0, 4.0, 2.8, 0.44, 6.7, 2.1]):
        new_feature_names.append(f'tan({name}/{divisor}*pi/3.0)')

    remove_list = []
    for i, name in enumerate(new_feature_names):
        # if 'tan' in name and any(tag in name for tag in ('VY', 'VTHETA')):
        #     remove_list.append(i)
        if any(tag in name for tag in ('VY', 'VTHETA')):
            remove_list.append(i)
    # print('-', len(remove_list))
    new_features = np.delete(new_features, remove_list, axis=0)
    new_feature_names = list(np.delete(new_feature_names, remove_list, axis=0))
    return new_features, new_feature_names

def add_custom_features(features, feature_names):
    ratios = np.divide(features[1], features[0])
    ratios = np.vstack((ratios, np.divide(features[2], features[0]+0.5)))
    ratios = np.vstack((ratios, np.divide(features[3], features[0]+0.5)))
    ratios = np.vstack((ratios, np.divide(features[4], features[0]+0.5)))
    ratios = np.vstack((ratios, np.divide(features[5], features[0]+0.5)))
    # ratios = np.vstack((ratios, np.divide(features[1], features[4])))
    # ratios = np.vstack((ratios, np.divide(features[2], features[4])))
    # ratios = np.vstack((ratios, np.divide(features[3], features[4])))
    # ratios = np.vstack((ratios, np.divide(features[5], features[4])))
    # ratios = np.vstack((ratios, np.divide(features[1], features[2])))
    # ratios = np.vstack((ratios, np.divide(features[2], features[1])))
    # ratios = np.vstack((ratios, np.divide(features[3], features[1])))
    # ratios = np.vstack((ratios, np.divide(features[0], features[5])))

    ratio_names = [
        f'{feature_names[1]}/({feature_names[0]}+0.5)',
        f'{feature_names[2]}/({feature_names[0]}+0.5)',
        f'{feature_names[3]}/({feature_names[0]}+0.5)',
        f'{feature_names[4]}/({feature_names[0]}+0.5)',
        f'{feature_names[5]}/({feature_names[0]}+0.5)',
        # f'{feature_names[2]}/{feature_names[0]}',
        # f'{feature_names[3]}/{feature_names[0]}',
        # f'{feature_names[4]}/{feature_names[0]}',
        # f'{feature_names[5]}/{feature_names[0]}',
        # f'{feature_names[1]}/{feature_names[4]}',
        # f'{feature_names[2]}/{feature_names[4]}',
        # f'{feature_names[3]}/{feature_names[4]}',
        # f'{feature_names[5]}/{feature_names[4]}',
        # f'{feature_names[1]}/{feature_names[2]}',
        # f'{feature_names[2]}/{feature_names[1]}',
        # f'{feature_names[3]}/{feature_names[1]}',
        # f'{feature_names[0]}/{feature_names[5]}',
    ]

    new_features = np.sin(ratios[0])
    new_features = np.vstack((new_features,np.cos(ratios[0])))
    new_features = np.vstack((new_features,np.tan(ratios[0])))
    for ratio in ratios[1:]:
        for trigo_fun in (np.sin, np.cos, np.tan):
            new_features = np.vstack((new_features, trigo_fun(ratio)))

    new_feature_names = []
    for ratio_name in ratio_names:
        for fun_name in ['sin', 'cos', 'tan']:
            new_feature_names.append(f'{fun_name}({ratio_name})')

    return new_features, new_feature_names

def add_constants(features, feature_names):
    new_features = np.ones((1,features.shape[1]))
    print(new_features)

if __name__ == '__main__':
    main()
