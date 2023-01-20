from evaluate_functions import compute_ndcg, compute_f1
import torch
from models import BaseRecModel
from args import *
import pickle

device = 'cpu'
dataset_path = "models\Dataset_20.pickle"
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

model_path = "models\model_20.model"
model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
model.load_state_dict(torch.load(model_path))
k = 5
f1_score = compute_f1(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            k, 
            model, 
            device)
ndcg_score = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            k, 
            model, 
            device)

print(f1_score)
print(ndcg_score)