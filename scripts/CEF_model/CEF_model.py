import torch
from ..base_model.models import BaseRecModel
from ..args import *
import pickle
import numpy as np
import tqdm
from collections import defaultdict

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
    

    def __init__(self, CEF_args, dataset, base_model, featurewise=False):
        super(CEF, self).__init__()
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

        self.dataset = dataset
        self.basemodel = base_model.to(self.device)
        
        self.args = CEF_args
        for p in self.basemodel.parameters():
            p.requires_grad = False
    
        
        
        if featurewise:
            self.delta_i = torch.zeros(self.dataset.item_feature_matrix.shape)
            self.delta_u = torch.zeros(self.dataset.user_feature_matrix.shape)
            self.deltadict_i = defaultdict(lambda: torch.nn.parameter.Parameter(torch.randn(self.dataset.item_num)/100))
            self.deltadict_u = defaultdict(lambda: torch.nn.parameter.Parameter(torch.randn(self.dataset.user_num)/100))
            self.params_i = None
            self.params_u = None
        else:
            self.delta_i = torch.nn.Parameter(torch.randn(self.dataset.item_feature_matrix.shape) / 1000).to(self.device)
            self.delta_u = torch.nn.Parameter(torch.randn(self.dataset.user_feature_matrix.shape) / 1000).to(self.device)
            


        self.update_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix,
                                                        k=5)

        self.exposure = dict()
        self.init_exposure = self.update_exposures(init=True)
        self.ranked_features = None


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

    def update_exposures(self, init=False):
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

        if init:
            return {"G0" : exposure_g0, "G1" : exposure_g1}


    def evaluate_model(self):
        self.update_exposures()
        ltr = self.exposure["G1"] / (self.exposure["G0"] + self.exposure["G1"])
        print(f"long tail rate: {ltr}")


    def loss_fn(self, disparity, ld):
        delta = torch.cat((self.delta_i, self.delta_u), dim=0)
        loss = disparity * disparity + ld * torch.linalg.norm(delta) / self.dataset.feature_num
        return loss

    def validity(self, feature_id, k=5):
        m = self.dataset.user_feature_matrix.shape[0]
        # TODO perturbation of only feature_id
        delta_i = self.delta_i.detach().numpy()
        delta_u = self.delta_u.detach().numpy()

        adjusted_if_matrix = self.dataset.item_feature_matrix.copy()
        adjusted_if_matrix[:,feature_id] += delta_i[:,feature_id]
        adjusted_uf_matrix = self.dataset.user_feature_matrix.copy()
        adjusted_uf_matrix[:,feature_id] += delta_u[:,feature_id]
        self.update_recommendations(adjusted_if_matrix, adjusted_uf_matrix)
        self.update_exposures()

        og_exp0 = self.init_exposure["G0"]
        og_exp1 = self.init_exposure["G1"]

        cf_exp0 = self.exposure["G0"]
        cf_exp1 = self.exposure["G1"]

        v = (og_exp0 - og_exp1 - cf_exp0 + cf_exp1) / (m * k)

        return v

    def top_k(self, beta=0.1):
        delta = torch.cat((self.delta_u.detach(), self.delta_i.detach()), dim=0).detach().numpy()
        ES_scores = {}
        for i in tqdm.trange(delta.shape[1]): #delta.shape[1]
            prox = np.linalg.norm(delta[:,i])**2
            validity = self.validity(i)

            ES_scores[i] = validity - beta * prox
        # sort ES_scores list
        ranked_features = [i[0] for i in sorted(ES_scores.items(), key = lambda item : item[1], reverse=True)]
        # self.ranked_features = ranked_features
        # Return top k features
        return ranked_features#[:5]




"""


import torch
import numpy as np
from ..evaluation.eval_model import *
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
    

    def __init__(self, CEF_args, dataset, base_model):
        super(CEF, self).__init__()
        self.device = CEF_args.device
        self.dataset= dataset
        self.basemodel= base_model
        self.k = CEF_args.rec_k
        for p in self.basemodel.parameters():
            p.requires_grad = False



        self.delta = torch.nn.Parameter(torch.randn(self.dataset.item_feature_matrix.shape)/100000)

        self.update_recommendations(self.dataset.item_feature_matrix, 
                                                        self.dataset.user_feature_matrix,
                                                        k=self.k)

        self.exposure = dict()
        self.init_exposure = self.update_exposures(init=True)


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

    def update_exposures(self, init=False):
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

        if init:
            return {"G0" : exposure_g0, "G1" : exposure_g1}


    def evaluate_model(self):
        self.update_exposures()
        #_, _, ltr = eval_model(self.dataset, 5, self.basemodel, self.device)
        ltr = self.exposure["G1"] / (self.exposure["G0"] + self.exposure["G1"])
        print(f"long tail rate: {ltr}")


    def loss_fn(self, disparity, ld, delta):
        loss = disparity * disparity + ld * torch.linalg.norm(delta) / self.dataset.feature_num
        return loss

    def validity(self,delta, feature_id, k=5):
        m = self.dataset.user_feature_matrix.shape[0]
        # TODO perturbation of only feature_id
        delta = delta.detach().numpy()

        adjusted_if_matrix = self.dataset.item_feature_matrix.copy()
        adjusted_uf_matrix = self.dataset.user_feature_matrix.copy()
        adjusted_if_matrix[:,feature_id] += delta[:,feature_id]
        self.update_recommendations(adjusted_if_matrix, adjusted_uf_matrix)
        self.update_exposures()

        og_exp0 = self.init_exposure["G0"]
        og_exp1 = self.init_exposure["G1"]

        cf_exp0 = self.exposure["G0"]
        cf_exp1 = self.exposure["G1"]

        v = (og_exp0 - og_exp1 - cf_exp0 + cf_exp1) / (m * k)

        return v

    def top_k(self, beta=0.1):
        delta = self.delta
        delta = delta.detach()
        ES_scores = {}
        for i in tqdm.trange(delta.shape[1]):
            prox = torch.norm(delta[:,i])
            validity = self.validity(delta, i)
            ES_scores[i] = validity - beta * prox
        # sort ES_scores list
        ranked_features = sorted(ES_scores.items(), key = lambda item : item[1], reverse=True)
        ranked_feature_ids = [i[0] for i in ranked_features]
        # Return top k features
        return ranked_feature_ids





    # def update_recommendations_cf(self, delta_v, delta_u):
    #     np.random.seed(42)
    #     dataset = self.dataset

    #     item_feature_matrix = dataset.item_feature_matrix + delta_v
    #     user_feature_matrix = dataset.user_feature_matrix + delta_u

    #     self.recommendations = self.get_recommendations(dataset.item_feature_matrix, dataset.user_feature_matrix, dataset.test_data)
    #     self.update_exposures()
    #     disparity = self.get_cf_disparity(self.recommendations)

    #     return disparity
"""