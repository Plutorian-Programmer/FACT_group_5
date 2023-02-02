import pickle
import numpy as np


with open('results/ndcg_lt_52_50__2_12.pickle', 'rb') as f:
    results = pickle.load(f)



for method in ['random', 'pop_item', 'pop_user', 'CEF']:
    for i in [1, 2, 4]:
        print(f"{method} after {i*50} feature deletions: ndcg = {np.round(results[method]['ndcg'][i], 5)*100}, lt = {np.round(results[method]['lt'][i], 5)*100}") # in percentages

    print("=====================================")

