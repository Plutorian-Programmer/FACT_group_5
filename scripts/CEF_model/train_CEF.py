import numpy as np
import torch
from .CEF_model import *

def train_delta(args, model):
    ld = args.ld
    lr = args.lr
    # Init values
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta_i, model.delta_u],lr=lr, betas=(0.9,0.999))

    # for i, p in enumerate(model.parameters()):
    #     if i > 0:
    #         p.requires_grad = False

    for i in tqdm.trange(args.epochs):
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
        print(f"Disparity: {disparity[0]}")
        print(f"loss: {loss[0]}")

        model.evaluate_model()

    # output_path = "models/CEF_model_full.model"
    # torch.save(model.state_dict(), output_path)
    torch.save(model.state_dict(), args.model_path)

    return model



if __name__ == "__main__":
    args = arg_parser_training()
    args.device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    with open('../../data/preprocessed_data/dataset.pickle', "rb") as f:
        dataset = pickle.load(f)
    base_model = BaseRecModel(dataset.feature_num, dataset).to(args.device)
    base_model.load_state_dict(torch.load('../../data/models/model.model'))
    model = CEF(args, dataset, base_model, featurewise=False)

    model = train_delta(model)
    # torch.save(delta_i, 'models/CEFout/delta_i_500features_full.pt')
    # torch.save(delta_u, 'models/CEFout/delta_u_500features_full.pt')
    # torch.save(model.state_dict(), 'models/CEF_model_500features_full.model')

    # ids_to_delete = model.top_k()
    # with open("models/CEFout/ids_500features_full.pickle", "wb") as f:
    #     pickle.dump(ids_to_delete, f)





"""
import numpy as np
import torch
from .CEF_model import *

def train_delta(CEF_args, model):
    # Init values
    lr = CEF_args.lr
    ld = CEF_args.ld
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta],lr=lr, betas=(0.9,0.999))

    for i, p in enumerate(model.parameters()):
        if i > 0:
            p.requires_grad = False

    for i in tqdm.trange(CEF_args.epochs):
        model.train()
        optimizer.zero_grad()

        adjusted_if_matrix = if_matrix.clone()
        adjusted_uf_matrix = uf_matrix.clone()
        adjusted_if_matrix[:,:] += model.delta
            
        model.update_recommendations(adjusted_if_matrix.detach().numpy(), adjusted_uf_matrix.detach().numpy(), delta=model.delta.clone().detach().numpy())
        disparity = model.get_cf_disparity(model.recommendations, adjusted_if_matrix, adjusted_uf_matrix)
        loss = model.loss_fn(disparity, ld, model.delta)

        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss.backward(retain_graph=True)

        print(model.delta.grad.norm())

        optimizer.step()

        print(f"epoch {i}")
        print(f"Disparity: {disparity}")
        print(f"loss: {loss}")

        model.evaluate_model()
    output_path = CEF_args.model_path
    torch.save(model.state_dict(), output_path)
    return model
"""