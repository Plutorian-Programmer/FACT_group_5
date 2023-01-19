import numpy as np

def get_feature_list(sentiment_data):
    """
    from user sentiment data, get all the features [F1, F2, ..., Fk] mentioned in the reviews
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: feature set F
    """
    feature_set = {}
    for row in sentiment_data:
        for fos in row[2:]:
            feature = fos[0]
            feature_set.add(feature)
    feature_list = np.array(list(feature_set))
    return feature_list

