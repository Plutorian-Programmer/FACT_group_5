from ..evaluation.eval_model import *
from ..utils import *
import pickle
from collections import defaultdict
import copy

def beta_testing(dataset, result_args, base_model, CEF_Model, beta_list = [0.1]):
    print("test")
    e = result_args.remove_size
    k = result_args.rec_k
    device = result_args.device

    feature_count = min(dataset.feature_num//e, result_args.epochs)
    beta_count = len(beta_list)

    results = defaultdict(lambda : { "ndcg" : [], "lt" : [] } )
    
    dataset_list = [copy.deepcopy(dataset) for _ in range(beta_count)]
    delete_lists = []
    # for beta in beta_list:
    #     CEF_delete_list = CEF_Model.top_k(beta)
    #     delete_lists.append(CEF_delete_list)
    data_path = "data/additional_work/delete_lists.pickle"
    # with open(data_path, "wb") as f:
    #     pickle.dump(delete_lists, f)
    print("start loading")
    with open(data_path, "rb") as f:
        delete_lists = pickle.load(f)
    
    print("file loaded")
    for idx in range(len(delete_lists)):
        delete_lists[idx] = delete_lists[idx][::-1]

    print("List reversed")
    ndcg, _ , lt = eval_model(dataset, k, base_model, device)
    for beta in beta_list:
        key = f"CEF_{beta}"
        results[key]["ndcg"].append(ndcg)
        results[key]["lt"].append(lt)
    
    progress = [i/10 for i in range(12)]
    for _ in range(feature_count):
        if progress != [] and _/feature_count >= progress[0]:
            print(progress.pop(0))
        #CEF
        for i in range(beta_count):
            CEF_delete_list = delete_lists[i]
            idx_list = CEF_delete_list[:e]
            CEF_delete_list = CEF_delete_list[e:]
            delete_lists[i] = CEF_delete_list

            CEF_dataset = dataset_list[i]
            CEF_dataset.remove_features(idx_list)
            dataset_list[i] = CEF_dataset
            ndcg_CEF, _ , lt_CEF = eval_model(CEF_dataset, k, base_model, device)
            
            beta = beta_list[i]
            key = f"CEF_{beta}"
            results[key]["ndcg"].append(ndcg_CEF)
            results[key]["lt"].append(lt_CEF)



    results = dict(results)
    output_path = result_args.output_path + f"results_{feature_count}_features_{e}_removals.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    
    return results