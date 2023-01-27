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

G0 = rec_dataset.G0
G1 = rec_dataset.G1
print(rec_dataset.test_data)