import torch
from sklearn.metrics import ndcg_score
import numpy as np

def compute_ndcg(test_data, user_feature_matrix, item_feature_matrix, k, model, device):
    model.eval()
    ndcgs = []
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            items = row[1]
            gt_labels = row[2]
            user_features = np.array([user_feature_matrix[user] for _ in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            ndcg = ndcg_score([gt_labels], [scores], k=k)
            ndcgs.append(ndcg)
    ave_ndcg = np.mean(ndcgs)
    return ave_ndcg

def compute_f1(test_data, user_feature_matrix, item_feature_matrix, k, model, device):
    model.eval()
    f1_scores = []
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            items = row[1]
            gt_labels = row[2]
            user_features = np.array([user_feature_matrix[user] for _ in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            # print(scores)
            # print(len(scores))
            pred_items = np.argsort(scores)[-k:]
            real_items = np.where(gt_labels == 1)[0]
            tp = 0
            fp = 0
            for item in pred_items:
                if item in real_items:
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
    ave_f1 = np.mean(f1_scores)
    return ave_f1

def compute_long_tail(test_data, user_feature_matrix, item_feature_matrix, k, model, device, G1):
    #my results are opposite of paper
    model.eval()
    long_tail_rates = []
    G0_exposure = 0
    G1_exposure = 0
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            items = row[1]
            user_features = np.array([user_feature_matrix[user] for _ in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            pred_items = np.argsort(scores)[-k:]
            LT_rate = 0
            for item in pred_items:
                if item in G1:
                    G1_exposure += 1
                else:
                    G0_exposure += 1
    return G1_exposure / (G1_exposure / G0_exposure)

def compute_KL(test_data, user_feature_matrix, item_feature_matrix, k, model, device, G0, G1):
    model.eval()
    kl_scores = []
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            items = row[1]
            user_features = np.array([user_feature_matrix[user] for _ in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            pred_items = np.argsort(scores)[-k:]
            true_dist = 0
            for item in pred_items:
                if item in G0:
                    true_dist += 1/k
            kl_score = 0
            for item in pred_items:
                if item in G0:
                    kl_score += true_dist*np.log(true_dist/0.5)
                else:
                    kl_score += (1-true_dist) * np.log((1-true_dist)/0.5)
            kl_scores.append(kl_score)
    return np.mean(kl_scores)