from utils import split_groups

class Group(object):
    exp = None
    items = None
    mag = None

    def __init__(self, sentiment_data, long_tail=False):
        self.exp = None
        if long_tail:
            self.items = split_groups(sentiment_data)[1]
        else:
            self.items = split_groups(sentiment_data)[0]
        
        self.mag = len(self.items)