import copy
import random
import numpy as np

def baseline_random(dataset):
    # return a shuffled list of idx to remove in order
    total_features = dataset.feature_num
    np.random.seed(42)
    removal_list = np.arange(total_features)
    np.random.shuffle(removal_list)

    return removal_list
    

def baseline_pop(dataset, method = "user"):
    # return a list of idx to remove in order based on how popular each feature is
    if method == "user":
        feature_matrix = dataset.user_feature_matrix
    else:
        feature_matrix = dataset.item_feature_matrix
    feature_matrix = np.where(feature_matrix != 0, 1, 0)
    existence_array = np.sum(feature_matrix, axis=0)
    existence_array = np.argsort(existence_array)[::-1]
    removal_list = existence_array
    return removal_list
