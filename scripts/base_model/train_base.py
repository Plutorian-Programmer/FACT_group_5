from .preprocessing import Dataset
from torch.utils.data import DataLoader
from ..dataloaders import UserItemInterDataset
from ..evaluation.eval_model import * #compute_ndcg, compute_f1
import torch
from .models import BaseRecModel
import tqdm
import numpy as np

def trainmodel(train_args, rec_dataset):
    train_args.gpu = False
    if train_args.gpu:
        device = torch.device('cuda')
    else:
        device = 'cpu'

    train_loader = DataLoader(dataset=UserItemInterDataset(rec_dataset.training_data, 
                            rec_dataset.user_feature_matrix, 
                            rec_dataset.item_feature_matrix),
                        batch_size=train_args.batch_size,
                        shuffle=True)

    model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr, weight_decay=1e-5)

    # Training loop
    for epoch in tqdm.trange(train_args.epochs): #train_args.epoch
        model.train()
        optimizer.zero_grad()
        losses = []
        for user_behaviour_feature, item_aspect_feature, label in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            item_aspect_feature = item_aspect_feature.to(device)
            label = label.float().to(device)
            out = model(user_behaviour_feature, item_aspect_feature).squeeze()
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.to('cpu').detach().numpy())
            ave_train = np.mean(np.array(losses))
        print('epoch %d: ' % epoch, 'training loss: ', ave_train)

        if epoch % 10 == 0:
            ndcg, _, _ = eval_model(rec_dataset, 5, model, device)
            print('epoch %d: ' % epoch, 'training loss: ', ave_train, 'NDCG: ', ndcg)
    
    output_path = train_args.model_path
    torch.save(model.state_dict(), output_path)
    return model