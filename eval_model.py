from evaluate_functions import *
from baselines import *
from utils import *
import torch
from models import BaseRecModel
from args import *
import pickle

device = 'cpu'
dataset_path = "models\Dataset.pickle"
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

model_path = "models\model.model"
model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
model.load_state_dict(torch.load(model_path))
k = 5

def run_tests(dataset, model):
    k = 5
    device = 'cpu'
    test_data = dataset.test_data
    user_feature_matrix = dataset.user_feature_matrix
    item_feature_matrix = dataset.item_feature_matrix 
    f1_score = compute_f1(test_data, 
                        user_feature_matrix, 
                        item_feature_matrix, 
                        k, 
                        model, 
                        device)
    ndcg_score = compute_ndcg(test_data, 
                            user_feature_matrix, 
                            item_feature_matrix, 
                            k, 
                            model, 
                            device)
    G0, G1 = get_groups(dataset)
    LT_rate = compute_long_tail(test_data, 
                                user_feature_matrix, 
                                item_feature_matrix, 
                                k, 
                                model, 
                                device, 
                                G1)
    KL = compute_KL(test_data, 
                            user_feature_matrix, 
                            item_feature_matrix, 
                            k, 
                            model, 
                            device,
                            G0, 
                            G1)
    print(f"F1: {f1_score}")
    print(f"NDCG: {ndcg_score}")
    print(f"LT_rate: {LT_rate}")
    print(F"KL: {KL}")



print("run default")
run_tests(rec_dataset, model)

print("run random")
random_dataset = baseline_random(rec_dataset)
run_tests(random_dataset, model)
print("run pop item")
pop_item_dataset = baseline_pop(rec_dataset, pop_method="item")
run_tests(pop_item_dataset, model)
print("run pop user")
pop_user_dataset = baseline_pop(rec_dataset, pop_method="user")
run_tests(pop_user_dataset, model)
# f1_score_random = compute_f1(rec_dataset.test_data, 
#             rec_dataset.user_feature_matrix, 
#             rec_dataset.item_feature_matrix, 
#             k, 
#             model, 
#             device)

# avg_f1_random = 0
# avg_ndcg_random = 0
# for i in range(10):

#     random_dataset = baseline_random(rec_dataset)

#     f1_score_random = compute_f1(rec_dataset.test_data, 
#                 random_dataset.user_feature_matrix, 
#                 random_dataset.item_feature_matrix, 
#                 k, 
#                 model, 
#                 device)
#     ndcg_score_random = compute_ndcg(rec_dataset.test_data, 
#                 random_dataset.user_feature_matrix, 
#                 random_dataset.item_feature_matrix, 
#                 k, 
#                 model, 
#                 device)
#     avg_f1_random += f1_score_random / 100
#     avg_ndcg_random += ndcg_score_random / 100

# print(f"F1_random: {avg_f1_random}")
# print(f"NDCG_random: {avg_ndcg_random}")

# pop_item_dataset = baseline_pop(rec_dataset, pop_method="item")
# f1_score_pop_item = compute_f1(rec_dataset.test_data, 
#             rec_dataset.user_feature_matrix, 
#             rec_dataset.item_feature_matrix, 
#             k, 
#             model, 
#             device)
# ndcg_score_pop_item = compute_ndcg(rec_dataset.test_data, 
#             rec_dataset.user_feature_matrix, 
#             rec_dataset.item_feature_matrix, 
#             k, 
#             model, 
#             device)

# print(f"F1 pop item: {f1_score_pop_item}")
# print(f"NDCG pop item: {ndcg_score_pop_item}")

# pop_user_dataset = baseline_pop(rec_dataset, pop_method="user")
# f1_score_pop_user = compute_f1(rec_dataset.test_data, 
#             rec_dataset.user_feature_matrix, 
#             rec_dataset.item_feature_matrix, 
#             k, 
#             model, 
#             device)
# ndcg_score_pop_user = compute_ndcg(rec_dataset.test_data, 
#             rec_dataset.user_feature_matrix, 
#             rec_dataset.item_feature_matrix, 
#             k, 
#             model, 
#             device)

# print(f"F1 pop user: {f1_score_pop_user}")
# print(f"NDCG pop user: {ndcg_score_pop_user}")