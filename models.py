import torch
import numpy as np


class BaseRecModel(torch.nn.Module):
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
        fusion = np.multiply(user_feature, item_feature)
        fusion = fusion.to(torch.float32)
        out = self.fc(fusion)
        return out


    
