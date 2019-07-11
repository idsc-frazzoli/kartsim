#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 14.06.19 10:05

@author: mvb
"""
import numpy as np
import random


def shuffle_dataframe(features, labels, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(labels.index)
    shuffled_features = features.reindex(idx)
    shuffled_labels = labels.reindex(idx)
    return shuffled_features, shuffled_labels


def shuffle_array(features, labels, random_seed=None):
    nr_of_data_points, nr_of_labels = labels.shape
    merged_data = np.hstack((features, labels))
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(merged_data)
    shuffled_features = merged_data[:, :-nr_of_labels]
    shuffled_labels = merged_data[:, -nr_of_labels:]
    return shuffled_features, shuffled_labels


def shuffle_list(features, labels, random_seed=None):
    merged_data = []
    for feature, label in zip(features, labels):
        merged_data.append([feature, label])

    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(merged_data)
    shuffled_features = []
    shuffled_labels = []
    for shuffled_feature, shuffled_label in merged_data:
        shuffled_features.append(shuffled_feature)
        shuffled_labels.append(shuffled_label)
    return np.array(shuffled_features), np.array(shuffled_labels)
