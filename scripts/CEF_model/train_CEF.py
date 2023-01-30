import numpy as np
import torch
from CEF_model import *

def train_delta(CEF_args, model):
    # Init values
    lr = CEF_args.lr
    ld = CEF_args.ld
    if_matrix = torch.Tensor(model.dataset.item_feature_matrix)
    uf_matrix = torch.Tensor(model.dataset.user_feature_matrix)
    optimizer = torch.optim.Adam([model.delta],lr=lr*1, betas=(0.9,0.999))

    for i, p in enumerate(model.parameters()):
        if i > 0:
            p.requires_grad = False

    for i in tqdm.trange(CEF_args.epochs):
        model.train()
        optimizer.zero_grad()

        adjusted_if_matrix = if_matrix.clone()
        adjusted_uf_matrix = uf_matrix.clone()
        adjusted_uf_matrix[:,:] += model.delta
            
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
