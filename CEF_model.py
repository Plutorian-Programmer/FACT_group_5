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
    recommendations = None
    device = None
    exposure = None
    delta = None
    

    def __init__(self):
        self.device = 'cpu'
        dataset_path="models/Dataset_20.pickle"
        model_path="models/model_20.model"

        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        
        self.basemodel = BaseRecModel(self.dataset.feature_num, self.dataset).to(self.device)
        self.basemodel.load_state_dict(torch.load(model_path))

        self.exposure = dict()

        self.update_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix,
                                                        k=5)

        self.delta = np.random.randn(self.dataset.item_feature_matrix.shape[1]) / 10
        self.params = [torch.nn.Parameter(torch.Tensor([d])) for d in self.delta]

    def update_recommendations(self, item_feature_matrix, user_feature_matrix, k=5):
        self.recommendations = {}
        test_data = self.dataset.test_data

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
        c = len(self.dataset.G0) / len(self.dataset.G1)
        for row in self.dataset.test_data:
            user = row[0]
            recommendations = self.recommendations[user]
            for item in recommendations:
                if item in self.dataset.G0:
                    disparity += self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))
                elif item in self.dataset.G1:
                    disparity -= c * self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))

                total += self.basemodel(torch.from_numpy(self.dataset.user_feature_matrix[user]), torch.from_numpy(self.dataset.item_feature_matrix[item]))

        disparity = disparity / total
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


    def loss_fn(self, disparity, ld, delta):
        return disparity**2 + ld * np.linalg.norm( np.array([param.detach() for param in delta], dtype='float32'))


    def train_delta(self):
        ld = 1
        lr = 0.01
        # Init values
        self.delta = np.random.randn(self.dataset.item_feature_matrix.shape[1]) / 10
        self.params = [torch.nn.Parameter(torch.Tensor([d])) for d in self.delta]

        adjusted_if_matrix = self.dataset.item_feature_matrix.copy()
        adjusted_uf_matrix = self.dataset.item_feature_matrix.copy()
        adjusted_if_matrix[0,:] += self.delta

        self.update_recommendations(adjusted_if_matrix, adjusted_uf_matrix)
        disparity = self.get_cf_disparity(self.recommendations)

        optimizer = torch.optim.Adam(self.params,lr=lr, betas=(0.9,0.999))
        loss_fn = self.loss_fn

        for i in range(10):
            optimizer.zero_grad()

            self.delta = np.array([param.detach() for param in self.params], dtype='float32')

            adjusted_if_matrix = self.dataset.item_feature_matrix.copy()
            adjusted_uf_matrix = self.dataset.item_feature_matrix.copy()
            adjusted_if_matrix[0,:] += self.delta
                
            self.update_recommendations(adjusted_if_matrix, adjusted_uf_matrix)
            disparity = self.get_cf_disparity(self.recommendations)
            
            loss = loss_fn(disparity, ld, self.params)
            loss.backward()
            optimizer.step()

            print(f"epoch {i}")
            print(f"Disparity: {disparity}")
            print(f"loss: {loss}")

            self.evaluate_model()



if __name__ == "__main__":
    CEF_model = CEF()
    CEF_model.evaluate_model()
    CEF_model.train_delta()




    # def update_recommendations_cf(self, delta_v, delta_u):
    #     np.random.seed(42)
    #     dataset = self.dataset

    #     item_feature_matrix = dataset.item_feature_matrix + delta_v
    #     user_feature_matrix = dataset.user_feature_matrix + delta_u

    #     self.recommendations = self.get_recommendations(dataset.item_feature_matrix, dataset.user_feature_matrix, dataset.test_data)
    #     self.update_exposures()
    #     disparity = self.get_cf_disparity(self.recommendations)

    #     return disparity
