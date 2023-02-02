from preprocessing import Dataset
from torch.utils.data import DataLoader
from dataloaders import UserItemInterDataset
from evaluate_functions import eval_model
import torch
from models import BaseRecModel
from args import *
import tqdm
import numpy as np
import os
import pickle

with open("models/Dataset.pickle", "rb") as f:
    dataset = pickle.load(f)

model = BaseRecModel(dataset.feature_num, dataset)
model.load_state_dict(torch.load("models/model.model"))

ndcg, f1, _ = eval_model(dataset, 5, model, 'cpu')
print(f"ndcg: {ndcg}")
print(f"f1: {f1}")