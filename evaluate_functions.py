import torch
from sklearn.metrics import ndcg_score
import numpy as np

def eval_model(dataset, k, model, device):
    model.eval()
    test_data = dataset.test_data
    user_feature_matrix = dataset.user_feature_matrix
    item_feature_matrix = dataset.item_feature_matrix
    G0 = dataset.G0
    G1 = dataset.G1
    ndcg_scores = []
    f1_scores = []
    lt_scores = []
    user_matrix = []
    item_matrix = []
    gt_list = []
    item_list = []
    for row in test_data:
        user = row[0]
        items = row[1]
        gt_labels = row[2]
        
        user_features = [user_feature_matrix[user] for _ in range(len(items))]
        item_features = [item_feature_matrix[item] for item in items]
        
        user_matrix += user_features
        item_matrix += item_features
        gt_list.append(gt_labels)
        item_list.append(items)

    user_matrix = np.array(user_matrix)
    item_matrix = np.array(item_matrix)
    with torch.no_grad():
        scores = model(torch.from_numpy(user_matrix).to(device),
                        torch.from_numpy(item_matrix).to(device)).squeeze()
        scores = np.array(scores.to('cpu'))
        counter = 0
        for idx, gt_labels in enumerate(gt_list):
            items = item_list[idx]
            total_items = len(gt_labels)
            user_scores = scores[counter : counter + total_items]
            counter += total_items
            # print(len(user_scores))
            # print(len(gt_labels))
            #ndcg
            ndcg = ndcg_score([gt_labels], [user_scores], k=k)
            ndcg_scores.append(ndcg)

            #f1
            pred_items = np.argsort(user_scores)[-k:]
            tp = 0
            fp = 0
            for item in pred_items:
                if gt_labels[item] == 1:
                    tp += 1
                else:
                    fp += 1
            if tp == 0:
                f1_scores.append(0)
                continue
            fn = k-tp
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            f1 = (2*prec*rec)/(prec+rec)
            f1_scores.append(f1)

            #lt
            lt = 0
            for item in pred_items:
                if items[item] in G1:
                    lt += 1/k
            lt_scores.append(lt)
    
    ndcg_scores = np.mean(ndcg_scores)
    f1_scores = np.mean(f1_scores)
    lt_scores = np.mean(lt_scores)

    return ndcg_scores, f1_scores, lt_scores

def compute_ltr(g0, g1):
    return g1 / (g1+g0)