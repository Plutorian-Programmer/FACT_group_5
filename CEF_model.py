import torch
from models import BaseRecModel
from args import *
import pickle
import numpy as np
from evaluate_functions import compute_ltr

# We define the following variables:
# R: the recommendation list
# g0: The set of popular items (20% of the items with the highest popularity)
# g1: The set of unpopular items (80% of the items with the lowest popularity)
# A: The user-feature matrix
# B: The item-feature matrix
# k: The number of items in the recommendation list

class CEF():
    dataset = None
    basemodel = None
    delta = None
    recommendations = None
    device = None
    exposure = None

    def __init__(self):
        self.device = 'cpu'
        dataset_path="models/Dataset_20.pickle"
        model_path="models/model_20.model"

        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        
        self.basemodel = BaseRecModel(self.dataset.feature_num, self.dataset).to(self.device)
        self.basemodel.load_state_dict(torch.load(model_path))

        self.exposure = dict()

        self.get_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix, 
                                                        self.dataset.test_data, 
                                                        k=5)

    def get_recommendations(self, item_feature_matrix, user_feature_matrix, test_data, k=5):
        self.recommendations = {}

        self.basemodel.eval()
        with torch.no_grad():
            for row in test_data:
                user = row[0]
                self.recommendations[user] = []
                items = row[1]
                gt_labels = row[2]
                user_features = np.array([user_feature_matrix[user] for i in range(len(items))])
                item_features = np.array([item_feature_matrix[item] for item in items])
                scores = self.basemodel(torch.from_numpy(user_features).to(self.device),
                                        torch.from_numpy(item_features).to(self.device)).squeeze() # het zijn 106 items ipv 105
                scores = np.array(scores.to('cpu'))
                indices = np.argsort(scores)[-k:] # indices (0-105) of items in the top k recs
                for i in indices:
                    self.recommendations[user].append(items[i]) # Get item IDs of top ranked items
 

    def get_cf_disparity(self, recommendations):
        disparity = 0
        total = 0
        c = self.dataset.G0.mag / self.dataset.G1.mag
        for row in self.dataset.test_data:
            user = row[0]
            recommendations = recommendations[user]
            for item in recommendations:
                if item in self.dataset.G0.items:
                    disparity += self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))
                elif item in self.dataset.G1.items:
                    disparity -= c * self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))

                total += self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))

        disparity = disparity / total
        return disparity

    def update_recommendations_cf(self, rec_dataset, item_feature_matrix, user_feature_matrix, test_data):
        np.random.seed(42)
        delta_v = np.random.randn(*item_feature_matrix.shape) / 10
        delta_u = np.random.randn(*user_feature_matrix.shape) / 10
        
        item_feature_matrix = rec_dataset.item_feature_matrix + delta_v
        user_feature_matrix = rec_dataset.user_feature_matrix + delta_u

        self.recommendations = self.get_recommendations(item_feature_matrix, user_feature_matrix, test_data)
        # rec_dataset = get_exposures(recs, rec_dataset) #update group exposures in rec_dataset
        disparity = self.get_cf_disparity(self.recommendations)

        return disparity

    def update_exposures(self):
        exposure_g0 = 0
        exposure_g1 = 0
        
        for recs in self.recommendations.values():
            for item in recs:
                if item in self.dataset.G0:
                    exposure_g0 += 1
                elif item in self.dataset.G1:
                    exposure_g1 += 1
        self.exposure["G0"] = exposure_g0
        self.exposure["G1"] = exposure_g1


    def evaluate_model(self):
        self.update_exposures()
        ltr = compute_ltr(self.exposure["G0"], self.exposure["G1"])
        print(f"long tail rate: {ltr}")


if __name__ == "__main__":
    CEF_model = CEF()
    CEF_model.evaluate_model()


