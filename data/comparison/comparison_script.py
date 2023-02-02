import torch
from CEF_model import CEF
from eval_model import run_tests
import pickle
from models import BaseRecModel
import matplotlib.pyplot as plt


delta_i_featurewise = torch.load('data/comparison/delta_i_500features_featurewise_Test.pt', map_location=torch.device("cpu"))
delta_u_featurewise = torch.load('data/comparison/delta_u_500features_featurewise_Test.pt', map_location=torch.device("cpu"))
delta_i_full = torch.load('data/comparison/delta_i_500features_full.pt', map_location=torch.device("cpu"))
delta_u_full = torch.load('data/comparison/delta_u_500features_full.pt', map_location=torch.device("cpu"))

with open("models/dataset_500_features.pickle", "rb") as f:
    dataset = pickle.load(f)

model = BaseRecModel(dataset.feature_num, dataset)
model.load_state_dict(torch.load("models/model_500_features.model")) 

model_CEF = CEF()
model_CEF.delta_u = torch.nn.parameter.Parameter(delta_u_featurewise)
model_CEF.delta_i = torch.nn.parameter.Parameter(delta_i_featurewise)
ranked_features_featurewise = model_CEF.top_k()
ranked_features_featurewise.reverse()
results_featurewise = run_tests(dataset, model, None, ranked_features_featurewise, mode="featurewise")

model_CEF.delta_u = torch.nn.parameter.Parameter(delta_u_full)
model_CEF.delta_i = torch.nn.parameter.Parameter(delta_i_full)
ranked_features_full = model_CEF.top_k()
ranked_features_full.reverse()
results_full = run_tests(dataset, model, None, ranked_features_full, mode="full")

def plot_results2(results_a, results_b):
    fwise = results_a["CEF"]
    full = results_b["CEF"]
    plt.plot(fwise["lt"], fwise["ndcg"], label="featurewise")
    plt.plot(full["lt"], full["ndcg"], label="full")

    plt.xlabel("long tail rate")
    plt.ylabel("NDCG")
    plt.legend()
    plt.show()
    return

with open("results/ndcg_lt_10_50_full.pickle", "rb") as f:
    results_full = pickle.load(f)
with open("results/ndcg_lt_10_50_featurewise.pickle", "rb") as f:
    results_featurewise = pickle.load(f)
plot_results2(results_featurewise, results_full)
