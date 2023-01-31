from .eval_model import *
from ..baselines import *
from ..utils import *
import torch
from ..base_model.models import BaseRecModel
from ..args import *
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
# from CEF_model import *
# from train_CEF import *
import copy

def get_results(dataset, result_args, base_model, CEF_Model, CEF_delete_list=None):
    np.random.seed(42)
    e = result_args.remove_size
    k = result_args.rec_k
    device = result_args.device
    if CEF_delete_list == None:
        CEF_delete_list = CEF_Model.top_k(result_args.beta)

    feature_count = min(dataset.feature_num//e, result_args.epochs)

    results = defaultdict(lambda : { "ndcg" : [], "lt" : [] } )
    
    random_dataset = baseline_random(dataset, 0)
    pop_item_dataset = baseline_pop(dataset, pop_method="item", e=0)
    pop_user_dataset = baseline_pop(dataset, pop_method="user", e=0)
    CEF_dataset = copy.deepcopy(dataset)
    visited_random = []

    ndcg, _ , lt = eval_model(dataset, k, base_model, device)
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
        ndcg_random, _ , lt_random = eval_model(random_dataset, k, base_model, device)
        results["random"]["ndcg"].append(ndcg_random)
        results["random"]["lt"].append(lt_random)

        #pop item
        pop_item_dataset = baseline_pop(pop_item_dataset, pop_method="item", e=e)
        ndcg_item, _ , lt_item = eval_model(pop_item_dataset, k, base_model, device)
        results["pop_item"]["ndcg"].append(ndcg_item)
        results["pop_item"]["lt"].append(lt_item)

        #pop user
        pop_user_dataset = baseline_pop(pop_user_dataset, pop_method="user", e=e)
        ndcg_user, _ , lt_user = eval_model(pop_user_dataset, k, base_model, device)
        results["pop_user"]["ndcg"].append(ndcg_user)
        results["pop_user"]["lt"].append(lt_user)

        #CEF
        removal_list = CEF_delete_list[:e]
        CEF_delete_list = CEF_delete_list[e:]
        CEF_dataset.remove_features(removal_list)
        ndcg_CEF, _ , lt_CEF = eval_model(CEF_dataset, k, base_model, device)
        results["CEF"]["ndcg"].append(ndcg_CEF)
        results["CEF"]["lt"].append(lt_CEF)

    results = dict(results)
    output_path = result_args.output_path + f"results_{feature_count}_features_{e}_removals.pickle"
    with open(output_path, "wb") as f:
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