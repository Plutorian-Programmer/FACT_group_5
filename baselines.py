from preprocessing import Dataset
import torch
from models import BaseRecModel
from args import *
import pickle
import copy
import random
import numpy as np

def baseline_random(dataset, e=5, method = "both", visited = []):
    dataset = copy.deepcopy(dataset)
    total_features = dataset.feature_num
    possible_features = []
    for feature in range(total_features):
        if feature not in visited:
            possible_features.append(feature)
    if len(possible_features) <= e:
        dataset.remove_features(possible_features, method)
    else:
        removal_list = random.sample(possible_features, e)
        dataset.remove_features(removal_list, method)
    visited += removal_list
    return dataset

def baseline_pop(dataset, pop_method = "user", e=5, method = "both"):
    dataset = copy.deepcopy(dataset)
    if pop_method == "user":
        feature_matrix = dataset.user_feature_matrix
    else:
        feature_matrix = dataset.item_feature_matrix
    feature_matrix = np.where(feature_matrix != 0, 1, 0)
    existence_array = np.sum(feature_matrix, axis=0)
    existence_array = np.argsort(existence_array)[::-1]
    removal_list = existence_array[:e]
    dataset.remove_features(removal_list, method)
    return dataset
