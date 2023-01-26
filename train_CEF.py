import numpy as np
import torch
from CEF_model import *

def train_delta(model):
    ld = 1
    lr = 0.01
    # Init values
    # model = CEF()
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta],lr=lr*1, betas=(0.9,0.999))

    # for i, p in enumerate(model.parameters()):
    #     if i > 0:
    #         p.requires_grad = False

    for i in tqdm.trange(1):
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

    return model.delta



if __name__ == "__main__":
    model = CEF()
    delta = train_delta(model)
    ids_to_delete = model.top_k(delta)



# def explainability_scores(self, feature, delta, k=5):
# #generate feature-based explanations based on updated exposures 
#     og_exp0 = self.og_exposure["G0"]
#     og_exp1 = self.og_exposure["G1"]
    
#     cf_exp0 = self.exposure["G0"]
#     cf_exp1 = self.exposure["G1"]

#     m = len(user_feature_matrix[0])     #number of users
#     # K = 20                              #length of recommendation lists
#     beta = 0.5

#     validity = ((og_exp0 - og_exp1) / (m * k))  - ((cf_exp0 - cf_exp1) / (m * k))
#     proximity = (np.linalg.norm(delta) **2)
#     ES = validity - beta * proximity


#     return(ES)