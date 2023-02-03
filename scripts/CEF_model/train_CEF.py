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

        optimizer.step()

        print(f"epoch {i}")
        print(f"Disparity: {np.round(disparity[0].detach().numpy(), 3)}")
        print(f"loss: {np.round(loss[0].detach().numpy(), 3)}")

        model.evaluate_model()

    output_path = args.model_path
    torch.save(model.state_dict(), output_path)
    return model