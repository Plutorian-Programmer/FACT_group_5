import torch
from models import BaseRecModel
from args import *
import pickle
import numpy as np

# We define the following variables:
# R: the recommendation list
# g0: The set of popular items (20% of the items with the highest popularity)
# g1: The set of unpopular items (80% of the items with the lowest popularity)
# A: The user-feature matrix
# B: The item-feature matrix
# k: The number of items in the recommendation list




# First we repeat "eval_model.py" to get A and B
device = 'cpu'
dataset_path = "models\Dataset_20.pickle"   #only 20
with open(dataset_path, "rb") as f:
    rec_dataset = pickle.load(f)

model_path = "models\model.model"
model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
model.load_state_dict(torch.load(model_path))
k = 5
# We define A and B with use of the user and item feature matrices from the preprocessing step.
A = rec_dataset.user_feature_matrix 
B = rec_dataset.item_feature_matrix
test_data = rec_dataset.test_data 

model.eval()
with torch.no_grad():
    for row in test_data:
        user = row[0]
        items = row[1]
        gt_labels = row[2]
        user_features = np.array([A[user] for i in range(len(items))])
        item_features = np.array([B[item] for item in items])


# We calculate the recommendation list R with k = 20
### Weet niet hoe ik dit moet doen  
# R = model(torch.tensor(user_features).float(), torch.tensor(item_features).float())
# R = R.cpu().numpy()
# R = np.argsort(R)[::-1][:k]

# We define U as the set of users that have interacted with at least one item in R
# U = []
# for user in R:
#     if user in U:
#         continue
#     else:
#         U.append(user)

 
 # exposure of recommendation model g is the number of items in R that g0 has interacted with and are in U
def exposure(g, R, U):
    exposure = 0
    for user in U:
        if user in R:
            exposure += 1
    return exposure


# calculate a quantification measure for diparity
def disparity(R, U):
    g0_exposure = exposure(0, R, U)
    g1_exposure = exposure(1, R, U)

    # take the difference between the two sides of the equalities as a quantification measure for disparity
    # theta_DP = abs(G1) * Exposure (G0 |R𝐾) − abs(G0) * Exposure (G1 |R𝐾)
    ##### weet niet hoe ik de absolute waarde van g0 moet nemen, is dat de totale lengte van de lijst?
    theta_DP = len(g1_exposure) * g0_exposure - len(g0_exposure) * g1_exposure
    return theta_DP

###Counterfactual reasoning
# for each user-feature vector A[:,f], we intervene with a vector delta_u (in R^m) to obtain a new user-feature vector A[:,f] + delta_u
for f in range(len(A)):
    for delta_u in range(len(A[f])):
        # we create a new user-feature vector A[:,f] + delta_u, A_cf
        A_cf = A[:,f] + delta_u
# for each item-feature vector B[:,f], we intervene with a vector delta_v (in R^m) to obtain a new item-feature vector B[:,f] + delta_v
for f in range(len(B)):
    for delta_v in range(len(B[f])):
        # we create a new item-feature vector B[:,f] + delta_v, B_cf
        B_cf = B[:,f] + delta_v


# we calculate the new recommendation list R_cf
# R_cf = R # weet niet hoe ik dit moet doen


# we take delta as the concatenation of delta_u and delta_v, delta = [delta_u, delta_v]
delta = [delta_u, delta_v]
# and we take hyper-parameter lambda as the weight of the counterfactual reasoning, between 0 and 1
l = 0.5

# We calculate the counterfactual fairness measure theta_CF
# theta_CF = l * disparity(R_cf, U) + (1 - l) * disparity(R, U)
