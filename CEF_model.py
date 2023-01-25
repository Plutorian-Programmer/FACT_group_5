import torch
from models import BaseRecModel
from args import *
import pickle
import numpy as np
from evaluate_functions import compute_ltr
import tqdm

# We define the following variables:
# R: the recommendation list
# g0: The set of popular items (20% of the items with the highest popularity)
# g1: The set of unpopular items (80% of the items with the lowest popularity)
# A: The user-feature matrix
# B: The item-feature matrix
# k: The number of items in the recommendation list
 



class CEF(torch.nn.Module):
    # dataset = None
    # basemodel = None
    # recommendations = None
    # device = None
    # exposure = None
    # delta = None
    

    def __init__(self):
        super(CEF, self).__init__()
        self.device = 'cpu'
        dataset_path="models/Dataset_20.pickle"
        model_path="models/model_20.model"

        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
        
        self.basemodel = BaseRecModel(self.dataset.feature_num, self.dataset).to(self.device)
        self.basemodel.load_state_dict(torch.load(model_path))

        self.exposure = dict()

        self.delta = torch.nn.Parameter(torch.randn(self.dataset.item_feature_matrix.shape[1]))

        self.update_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix,
                                                        k=5)


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
 

    def get_cf_disparity(self, recommendations, if_matrix, uf_matrix):
        disparity = 0
        total = 0
        c = len(self.dataset.G0) / len(self.dataset.G1)
        for row in self.dataset.test_data:
            user = row[0]
            recommendations = self.recommendations[user]
            self.basemodel.eval()
            for item in recommendations:
                if item in self.dataset.G0:
                    x = self.basemodel(uf_matrix[user],if_matrix[item])
                    disparity += x
                    total += x
                elif item in self.dataset.G1:
                    x = c * self.basemodel(uf_matrix[user], if_matrix[item])
                    disparity -= x
                    total += x

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
        return disparity**2 + ld * torch.linalg.norm(delta)


def train_delta():
    ld = 1
    lr = 0.01
    # Init values
    model = CEF()
    # print(list(model.parameters()))
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix.copy())
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix.copy())
    adjusted_if_matrix = if_matrix.clone().detach()
    adjusted_uf_matrix = uf_matrix.clone().detach()
    adjusted_if_matrix[0,:] += model.delta

    model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy())
    disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)

    # model.delta.retain_grad = True
    optimizer = torch.optim.Adam([model.delta],lr=lr*10, betas=(0.9,0.999))
    # loss_fn = model.loss_fn

    for i in tqdm.trange(30):
        model.train()
        optimizer.zero_grad()

        adjusted_if_matrix = if_matrix.clone().detach()
        adjusted_uf_matrix = uf_matrix.clone().detach()
        adjusted_if_matrix[0,:] += model.delta
            
        model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy())
        disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)
        
        # loss = loss_fn(disparity, ld, model.delta)
        loss = model.loss_fn(disparity, ld, model.delta)
        loss.backward()
        print(model.delta.grad.norm())
        optimizer.step()

        print(f"epoch {i}")
        print(f"Disparity: {disparity}")
        print(f"loss: {loss}")

        model.evaluate_model()



if __name__ == "__main__":
    # CEF_model = CEF()
    # CEF_model.evaluate_model()
    train_delta()




    # def update_recommendations_cf(self, delta_v, delta_u):
    #     np.random.seed(42)
    #     dataset = self.dataset

    #     item_feature_matrix = dataset.item_feature_matrix + delta_v
    #     user_feature_matrix = dataset.user_feature_matrix + delta_u

    #     self.recommendations = self.get_recommendations(dataset.item_feature_matrix, dataset.user_feature_matrix, dataset.test_data)
    #     self.update_exposures()
    #     disparity = self.get_cf_disparity(self.recommendations)

    #     return disparity
