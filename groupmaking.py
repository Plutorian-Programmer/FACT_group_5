
"""""function to be used in the train-base at the preprocessing step: using the get_user_item_dict from the preprocessing as well to obtain a user/item dictionary from which we can read the most popular items"""

#number of reviews for an item --> top 20% vs. bottom 80%



def split_groups(sentiment_data):
    user_dict, item_dict = sentiment_data.get_user_item_dict #method from preprocessing return that returns: user dictionary {u1:[i, i, i...], u2:[i, i, i...]}, similarly, item dictionary --> we use the item dict
    twenty_percent = round(0.2 * len(item_dict.keys()))
    first_20p_items = (dict(sorted(item_dict, key = lambda key: len(item_dict[key]), reverse=True)[:twenty_percent])).keys() #sort the dictionary by the value list length: so sort items by the number of reviews they got
    last_80p_items = (dict(sorted(item_dict, key = lambda key: len(item_dict[key]), reverse=True)[twenty_percent:])).keys()

    return(first_20p_items, last_80p_items)

    #this might be a lot of computational effort, because of the 2 times making the dict, looking for a more efficient way to do it. 


