from args import arg_parser_preprocessing
from utils import get_feature_list
import numpy as np
from collections import defaultdict
import json

class Dataset():
    def __init__(self, preprocessing_args):
        super().__init__()
        self.args = preprocessing_args

        self.sentiment_data = None
        self.user_name_dict = {}  # rename users to integer names
        self.item_name_dict = {}
        self.feature_name_dict = {}

        self.features = []  # feature list
        self.users = []
        self.items = []

        # the interacted items for each user, sorted with date {user:[i1, i2, i3, ...], user:[i1, i2, i3, ...]}
        self.user_hist_inter_dict = {}
        # the interacted users for each item
        self.item_hist_inter_dict = {} 

        self.user_num = None
        self.item_num = None
        self.feature_num = None  # number of features

        self.user_feature_matrix = None  # user aspect attention matrix
        self.item_feature_matrix = None  # item aspect quality matrix

        self.training_data = None
        self.test_data = None

        self.pre_processing()
        self.get_user_item_feature_matrix()
        self.sample_training()  # sample training data, for traning BPR loss
        self.sample_test()  # sample test data

    def pre_processing(self,):
        sentiment_data = []
        with open(self.args.sentires_dir, 'r') as f:
            line = f.readline().strip()
            while line:
                splitline = line.split('@')
                user = splitline[0]
                item = splitline[1]
                sentiment_data.append([user, item])
                fosr_data = splitline[3]
                for seg in fosr_data.split('||'):
                    fos = seg.split(':')[0].strip('|')
                    if len(fos.split('|')) > 1:
                        feature = fos.split('|')[0]
                        opinion = fos.split('|')[1]
                        sentiment = fos.split('|')[2]
                        sentiment_data[-1].append([feature, opinion, sentiment])
                line = f.readline().strip()
        sentiment_data = np.array(sentiment_data)
        # sentiment_data = sentiment_data_filtering(
        ###TODO Add sentiment data filtering (>=20 reviews per item and human)
        # )
        user_dict, item_dict = get_user_item_dict(sentiment_data)
        user_item_date_dict = {}

        # remove duplicates [could be optimized]
        for i, line in enumerate(open(self.args.review_dir, "r")):
            record = json.loads(line)
            user = record['reviewerID']
            item = record['asin']
            date = record['unixReviewTime']
            if user in user_dict and item in user_dict[user] and (user, item) not in user_item_date_dict:
                user_item_date_dict[(user, item)] = date

        user_name_dict = {}
        item_name_dict = {}
        feature_name_dict = {}
        features = get_feature_list(sentiment_data)
        


def get_user_item_dict(sentiment_data):
    """
    build user & item dictionary
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: user dictionary {u1:[i, i, i...], u2:[i, i, i...]}, similarly, item dictionary
    """
    user_dict = {}
    item_dict = {}
    for row in sentiment_data:
        user = row[0]
        item = row[1]
        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
        if item not in item_dict:
            item_dict[item] = [user]
        else:
            item_dict[item].append(user)
    return user_dict, item_dict


def preprocessing(preprocessing_args):
    r = Dataset(preprocessing_args)
    
if __name__ == "__main__":
    preprocessing_args = arg_parser_preprocessing()
    preprocessing(preprocessing_args)