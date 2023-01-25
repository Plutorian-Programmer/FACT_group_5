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
        for p in self.basemodel.parameters():
            p.requires_grad = False

        self.exposure = dict()

        self.delta = torch.nn.Parameter(torch.randn(self.dataset.item_feature_matrix.shape))

        self.update_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix,
                                                        k=5)


    def update_recommendations(self, item_feature_matrix, user_feature_matrix, delta=0, k=5):
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
                # if_matrix[item, 0] += delta[item]
                if item in self.dataset.G0:
                    x = self.basemodel(uf_matrix[user,:],if_matrix[item,:])
                    disparity += x
                    total += x
                elif item in self.dataset.G1:
                    x = c * self.basemodel(uf_matrix[user,:], if_matrix[item,:])
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
        loss = disparity * disparity #+ ld * torch.linalg.norm(delta)
        return loss


def train_delta():
    ld = 1
    lr = 0.01
    # Init values
    model = CEF()
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta],lr=lr*1, betas=(0.9,0.999))

    # for i, p in enumerate(model.parameters()):
    #     if i > 0:
    #         p.requires_grad = False

    for i in tqdm.trange(50):
        model.train()
        optimizer.zero_grad()

        adjusted_if_matrix = if_matrix.clone()
        adjusted_uf_matrix = uf_matrix.clone()
        adjusted_if_matrix[:,:] += model.delta
            
        model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy(), delta=model.delta.clone().detach().numpy())
        disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)
        loss = model.loss_fn(disparity, ld, model.delta)

        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss.backward(retain_graph=True)

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
