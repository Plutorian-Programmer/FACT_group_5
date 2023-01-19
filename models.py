import torch
import numpy as np
from preprocessing import Dataset
from torch.utils.data import DataLoader
from dataloaders import UserItemInterDataset
from evaluate_functions import compute_ndcg

class BaseRecModel(torch.nn.module):
    def __init__(self, feature_length, rec_dataset):
        super(BaseRecModel, self).__init__()

        self.rec_dataset = rec_dataset

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_length, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user_feature, item_feature):
        fusion = np.mutliply(user_feature, item_feature)
        out = self.fc(fusion)
        return out



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
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)

    ndcg = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
    print('init ndcg:', ndcg)



    
