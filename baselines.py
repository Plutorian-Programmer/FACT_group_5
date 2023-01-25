from preprocessing import Dataset
import torch
from models import BaseRecModel
from args import *
import pickle
import copy
import random
import numpy as np

def baseline_random(dataset, e=5, method = "both"):
    dataset = copy.deepcopy(dataset)
    removal_list = random.sample(range(0,dataset.feature_num), e)
    dataset.remove_features(removal_list, method)
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
