from preprocessing import Dataset
from baselines import *
import torch
from models import BaseRecModel
from args import *
import pickle
import copy
import random
import numpy as np

device = 'cpu'
dataset_path = "models\Dataset.pickle"
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

# model_path = "models\model.model"
# model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
# model.load_state_dict(torch.load(model_path))

user_matrix = rec_dataset.user_feature_matrix
item_matrix = rec_dataset.item_feature_matrix
feature_list = rec_dataset.features
print(len(feature_list))
user_matrix = np.where(user_matrix != 0, 1, 0)
existence_array = np.sum(user_matrix, axis=0)
sorted_existence_array = np.argsort(existence_array)[::-1]
print(existence_array[sorted_existence_array[0]])
print(existence_array[sorted_existence_array[1]])