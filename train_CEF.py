import numpy as np
import torch
from CEF_model import *

def train_delta(model):
    ld = 1
    lr = 0.01
    # Init values
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta_i, model.delta_u],lr=lr, betas=(0.9,0.999))

    # for i, p in enumerate(model.parameters()):
    #     if i > 0:
    #         p.requires_grad = False

    for i in tqdm.trange(100):
        model.train()
        optimizer.zero_grad()

        adjusted_if_matrix = if_matrix.clone()
        adjusted_uf_matrix = uf_matrix.clone()
        adjusted_if_matrix[:,:] += model.delta_i
        adjusted_uf_matrix[:,:] += model.delta_u
            
        model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy())
        disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)
        loss = model.loss_fn(disparity, ld)

        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss.backward(retain_graph=True)

        # print(f"delta grad size: {model.delta_i.grad.norm()}")

        optimizer.step()

        print(f"epoch {i}")
        print(f"Disparity: {np.round(disparity[0].detach().numpy(), 3)}")
        print(f"loss: {np.round(loss[0].detach().numpy(), 3)}")

        model.evaluate_model()

    # output_path = "models/CEF_model_full.model"
    # torch.save(model.state_dict(), output_path)
    return model.delta_i, model.delta_u, model



if __name__ == "__main__":
    model = CEF()
    
    model.update_exposures()
    print(model.exposure)

    delta_i, delta_u, model = train_delta(model)
    # torch.save(delta_i, 'models/CEFout/delta_i_500features_full.pt')
    # torch.save(delta_u, 'models/CEFout/delta_u_500features_full.pt')
    torch.save(model.state_dict(), 'models/CEF_model.model')

    ids_to_delete = model.top_k()
    with open("models/CEFout/ids_full.pickle", "wb") as f:
        pickle.dump(ids_to_delete, f)



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
