from evaluate_functions import *
from baselines import *
from utils import *
import torch
from models import BaseRecModel
from args import *
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from CEF_model import *
from train_CEF import *
import copy
device = 'cpu'

dataset_path = "models/Dataset_20_test.pickle"
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

model_path = "models/model_20_test.model"
model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
model.load_state_dict(torch.load(model_path))
k = 5


cef_model_path = "models/CEF_model_temp.model"
train_args = arg_parser_CEF()
cef_model = CEF(train_args)
cef_model.load_state_dict(torch.load(cef_model_path), strict=False)

with open(f"models/ids_temp.pickle", "rb") as f:
    CEF_ids_to_delete = pickle.load(f)

def run_tests(dataset, model, CEF_model, ids_to_delete):
    np.random.seed(42)
    e = 50
    k = 5
    device = "cpu"
    # CEF_model = CEF()
    delta = CEF_model.delta
    # ids_to_delete = CEF_model.top_k(delta)

    feature_count = min(dataset.feature_num//e, 1000)

    results = defaultdict(lambda : { "ndcg" : [], "lt" : [] } )
    
    random_dataset = baseline_random(dataset, 0)
    pop_item_dataset = baseline_pop(dataset, pop_method="item", e=0)
    pop_user_dataset = baseline_pop(dataset, pop_method="user", e=0)
    CEF_dataset = copy.deepcopy(dataset)
    visited_random = []

    ndcg, _ , lt = eval_model(dataset, k, model, device)
    results["random"]["ndcg"].append(ndcg)
    results["random"]["lt"].append(lt)
    results["pop_item"]["ndcg"].append(ndcg)
    results["pop_item"]["lt"].append(lt)
    results["pop_user"]["ndcg"].append(ndcg)
    results["pop_user"]["lt"].append(lt)
    results["CEF"]["ndcg"].append(ndcg)
    results["CEF"]["lt"].append(lt)
    
    progress = [i/10 for i in range(12)]
    for _ in range(feature_count):
        if progress != [] and _/feature_count >= progress[0]:
            print(progress.pop(0))
        # print(_)
        #random
        random_dataset = baseline_random(random_dataset, e=e, visited=visited_random)
        ndcg_random, _ , lt_random = eval_model(random_dataset, k, model, device)
        results["random"]["ndcg"].append(ndcg_random)
        results["random"]["lt"].append(lt_random)

        #pop item
        pop_item_dataset = baseline_pop(pop_item_dataset, pop_method="item", e=e)
        ndcg_item, _ , lt_item = eval_model(pop_item_dataset, k, model, device)
        results["pop_item"]["ndcg"].append(ndcg_item)
        results["pop_item"]["lt"].append(lt_item)

        #pop user
        pop_user_dataset = baseline_pop(pop_user_dataset, pop_method="user", e=e)
        ndcg_user, _ , lt_user = eval_model(pop_user_dataset, k, model, device)
        results["pop_user"]["ndcg"].append(ndcg_user)
        results["pop_user"]["lt"].append(lt_user)

        #CEF
        removal_list = ids_to_delete[:e]
        ids_to_delete = ids_to_delete[e:]
        CEF_dataset.remove_features(removal_list)
        ndcg_CEF, _ , lt_CEF = eval_model(CEF_dataset, k, model, device)
        results["CEF"]["ndcg"].append(ndcg_CEF)
        results["CEF"]["lt"].append(lt_CEF)

    results = dict(results)
    with open(f"results/ndcg_lt_{feature_count}_{e}.pickle", "wb") as f:
        pickle.dump(results, f)
    
    return results


def plot_results(results):
    for method in results:
        result = results[method]
        plt.scatter(result["lt"], result["ndcg"], label = method)
    
    plt.xlabel("long tail rate")
    plt.ylabel("NDCG")
    plt.legend()
    plt.show()

results = run_tests(rec_dataset, model, cef_model, CEF_ids_to_delete)

# with open("results/ndcg_lt_20.pickle", "rb") as f:
#     results = pickle.load(f)
plot_results(results)
# print(rec_dataset.feature_num)
# feature_count_list = []
# feature_matrix = rec_dataset.user_feature_matrix
# feature_matrix = np.where(feature_matrix != 0, 1, 0)
# existence_array = np.sum(feature_matrix, axis=0)
# existence_array = np.sort(existence_array)

# plt.plot(existence_array)
# plt.show()
# print(len(ids_to_delete))