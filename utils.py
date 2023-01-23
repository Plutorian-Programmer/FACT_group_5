import numpy as np
from collections import defaultdict
def get_feature_list(sentiment_data):
    """
    from user sentiment data, get all the features [F1, F2, ..., Fk] mentioned in the reviews
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: feature set F
    """
    feature_set = set()
    for row in sentiment_data:
        for fos in row[2:]:
            feature = fos[0]
            feature_set.add(feature)
    feature_list = np.array(list(feature_set))
    return feature_list

def sentiment_data_filtering(sentiment_data, item_tresh, user_tresh):
    user_dict, item_dict = get_user_item_dict(sentiment_data)
    features = get_feature_list(sentiment_data)
    print("original review length: ", len(sentiment_data))
    print("original user length: ", len(user_dict))
    print("original item length: ", len(item_dict))
    print("original feature length: ", len(features))
    print("-"*69)
    #sentiment_data = [[used_id, item_id, [feature, opinion, score]]]
    item_count = defaultdict(lambda: 0)
    for review in sentiment_data:
        item_count[review[1]] += 1
    
    allowed_items = [item for item in item_count.keys() if item_count[item] >= item_tresh]

    user_count = defaultdict(lambda: 0)
    for review in sentiment_data:
        if review[1] in allowed_items:
            user_count[review[0]] += 1
    
    allowed_users = [user for user in user_count.keys() if user_count[user] >= user_tresh]

    sentiment_data = [review for review in sentiment_data if review[0] in allowed_users and review[1] in allowed_items]
    
    user_dict, item_dict = get_user_item_dict(sentiment_data)
    features = get_feature_list(sentiment_data)
    print('valid review length: ', len(sentiment_data))
    print("valid user: ", len(user_dict))
    print('valid item : ', len(item_dict))
    print("valid feature length: ", len(features))
    print('user dense is:', len(sentiment_data) / len(user_dict))
    sentiment_data = np.array(sentiment_data)
    return sentiment_data

def get_user_attention_matrix(sentiment_data, user_num, feature_list, max_range=5):
    """
    build user attention matrix
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :param user_num: number of users
    :param feature_list: [F1, F2, ..., Fk]
    :param max_range: normalize the attention value to [1, max_range]
    :return: the user attention matrix, Xij is user i's attention on feature j
    """
    user_counting_matrix = np.zeros((user_num, len(feature_list)))  # tij = x if user i mention feature j x times
    for row in sentiment_data:
        user = row[0]
        for fos in row[2:]:
            feature = fos[0]
            user_counting_matrix[user, feature] += 1
    user_attention_matrix = np.zeros((user_num, len(feature_list)))  # xij = [1-N], normalized attention matrix
    for i in range(len(user_counting_matrix)):
        for j in range(len(user_counting_matrix[i])):
            if user_counting_matrix[i, j] == 0:
                norm_v = 0  # if nor mentioned: 0
            else:
                norm_v = 1 + (max_range - 1) * ((2 / (1 + np.exp(-user_counting_matrix[i, j]))) - 1)  # norm score
            user_attention_matrix[i, j] = norm_v
    user_attention_matrix = np.array(user_attention_matrix, dtype='float32')
    return user_attention_matrix


def get_item_quality_matrix(sentiment_data, item_num, feature_list, max_range=5):
    """
    build item quality matrix
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :param item_num: number of items
    :param feature_list: [F1, F2, ..., Fk]
    :param max_range: normalize the quality value to [1, max_range]
    :return: the item quality matrix, Yij is item i's quality on feature j
    """
    item_counting_matrix = np.zeros((item_num, len(feature_list)))  # kij = x if item i's feature j is mentioned x times
    item_sentiment_matrix = np.zeros((item_num, len(feature_list)))  # sij = x if the overall rating is x (sum up)
    for row in sentiment_data:
        item = row[1]
        for fos in row[2:]:
            feature = fos[0]
            sentiment = fos[2]
            item_counting_matrix[item, feature] += 1
            if sentiment == '+1':
                item_sentiment_matrix[item, feature] += 1
            elif sentiment == '-1':
                item_sentiment_matrix[item, feature] -= 1
            else:
                print("sentiment data error: the sentiment value can only be +1 or -1")
                exit(1)
    item_quality_matrix = np.zeros((item_num, len(feature_list)))
    for i in range(len(item_counting_matrix)):
        for j in range(len(item_counting_matrix[i])):
            if item_counting_matrix[i, j] == 0:
                norm_v = 0  # if not mentioned: 0
            else:
                norm_v = 1 + ((max_range - 1) / (1 + np.exp(-item_sentiment_matrix[i, j])))  # norm score
            item_quality_matrix[i, j] = norm_v
    item_quality_matrix = np.array(item_quality_matrix, dtype='float32')
    return item_quality_matrix

def sample_training_pairs(user, training_items, item_set, sample_ratio=10):
    positive_items = set(training_items)
    negative_items = set()
    for item in item_set:
        if item not in positive_items:
            negative_items.add(item)
    neg_length = len(positive_items) * sample_ratio
    negative_items = np.random.choice(np.array(list(negative_items)), neg_length, replace=False)
    train_pairs = []
    for p_item in positive_items:
        train_pairs.append([user, p_item, 1])
    for n_item in negative_items:
        train_pairs.append([user, n_item, 0])
    return train_pairs

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

def get_groups(dataset):
    interaction_count = np.zeros(dataset.item_num)
    for user in dataset.user_hist_inter_dict:
        for item in dataset.user_hist_inter_dict[user]:
            interaction_count[item] += 1

    sorted_items = np.argsort(interaction_count)[::-1]
    split = int(len(sorted_items)*0.2)
    G0 = list(sorted_items[:split])
    G1 = list(sorted_items[split:])
    return G0, G1