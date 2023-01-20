from preprocessing import Dataset
from torch.utils.data import DataLoader
from dataloaders import UserItemInterDataset
from evaluate_functions import compute_ndcg, compute_f1
import torch
from models import BaseRecModel
from args import *
import tqdm
import numpy as np
import os
import pickle

device = 'cpu'
dataset_path = "models\Dataset.pickle"
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

model_path = "models\model.model"
model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
model.load_state_dict(torch.load(model_path))

compute_f1(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            5, 
            model, 
            device)