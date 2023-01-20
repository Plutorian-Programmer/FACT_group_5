from preprocessing import Dataset
from torch.utils.data import DataLoader
from dataloaders import UserItemInterDataset
from evaluate_functions import compute_ndcg
import torch
from models import BaseRecModel
from args import *
import tqdm
import numpy as np
import os

def trainmodel(train_args, pre_processing_args):
    train_args.gpu = False
    if train_args.gpu:
        device = torch.device('cuda')
    else:
        device = 'cpu'

    rec_dataset = Dataset(pre_processing_args)

    train_loader = DataLoader(dataset=UserItemInterDataset(rec_dataset.training_data, 
                            rec_dataset.user_feature_matrix, 
                            rec_dataset.item_feature_matrix),
                        batch_size=train_args.batch_size,
                        shuffle=True)

    model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr)

    ndcg = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
    print('init ndcg:', ndcg)

    # Training loop
    for epoch in tqdm.trange(100): #train_args.epoch
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
            ndcg = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
            print('epoch %d: ' % epoch, 'training loss: ', ave_train, 'NDCG: ', ndcg)

    #torch.save(model.state_dict(), os.path.join(out_path, "model.model"))
    return 0


if __name__ == "__main__":
    train_args = arg_parser_training()
    pre_processing_args = arg_parser_preprocessing()
    trainmodel(train_args, pre_processing_args)