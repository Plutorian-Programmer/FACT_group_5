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
    with torch.no_grad():
        #loop over all test data
        for row in test_data:
            user = row[0]
            items = row[1]
            gt_labels = row[2]

            #get the prediction list for each user
            user_features = np.array([user_feature_matrix[user] for _ in range(len(items))])
            item_features = np.array([item_feature_matrix[item] for item in items])
            scores = model(torch.from_numpy(user_features).to(device),
                                    torch.from_numpy(item_features).to(device)).squeeze()
            scores = np.array(scores.to('cpu'))
            
            #ndcg
            ndcg = ndcg_score([gt_labels], [scores], k=k)
            ndcg_scores.append(ndcg)

            #f1
            pred_items = np.argsort(scores)[-k:]
            tp = 0
            fp = 0
            for item in pred_items:
                if gt_labels[item] == 1:
                    tp += 1
                else:
                    fp += 1
            if tp == 0:
                f1_scores.append(0)
            else:
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