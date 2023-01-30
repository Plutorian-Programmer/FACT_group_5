import numpy as np
import torch
from CEF_model import *

def train_delta(model):
    ld = model.args.ld
    lr = model.args.lr
    # Init values

    device = model.device
    model = model.to(device)

    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)

    # for i, p in enumerate(model.parameters()):
    #     if i > 0:
    #         p.requires_grad = False
    for feature in tqdm.trange(10): #model.dataset.feature_num
        print(f"feature {feature}")
        model.params = model.deltadict[feature].to(device)
        optimizer = torch.optim.Adam([model.params],lr=lr*1, betas=(0.9,0.999))
        for i in tqdm.trange(model.args.epochs):
            print(f"epoch {i}")
            model.train()
            optimizer.zero_grad()

            adjusted_if_matrix = if_matrix.clone().to(device)
            adjusted_uf_matrix = uf_matrix.clone().to(device)
            adjusted_if_matrix[:,feature] += model.params #elta[:,feature]
                
            model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy(), delta=model.delta.clone().detach().numpy())
            disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)
            loss = model.loss_fn(disparity, ld, model.params).to(device)

            # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            loss.backward(retain_graph=True)

            optimizer.step()
            model.delta[:,feature] = model.params
            print(f"Disparity: {disparity[0]}")
            print(f"loss: {loss[0]}")

            model.evaluate_model()
            
       

    return model.delta



if __name__ == "__main__":
    train_args = arg_parser_CEF()
    model = CEF(train_args)
    delta = train_delta(model)
    # torch.save(delta, 'models/delta.pt')
    torch.save(model.state_dict(), f'models/CEF_model_temp.model')

    ids_to_delete = model.top_k(delta)
    with open(f"models/ids_temp.pickle", "wb") as f:
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
