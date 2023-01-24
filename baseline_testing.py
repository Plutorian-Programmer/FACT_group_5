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

interaction_count = np.zeros(rec_dataset.item_num)
for user in rec_dataset.user_hist_inter_dict:
    for item in rec_dataset.user_hist_inter_dict[user]:
        interaction_count[item] += 1

sorted_items = np.argsort(interaction_count)[::-1]
split = int(len(sorted_items)*0.2)
top = list(sorted_items[:split])
bottom = list(sorted_items[split:])
print(len(top))
print(len(bottom))

top_sum = 0
for item in top:
    top_sum += interaction_count[item]
bottom_sum = np.sum(interaction_count) - top_sum

print(top_sum)
print(bottom_sum)
print(interaction_count[top[0]])
print(interaction_count[top[-1]])
print(interaction_count[bottom[0]])
print(interaction_count[bottom[-1]])