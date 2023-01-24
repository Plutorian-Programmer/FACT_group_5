# from CEF_model import *


# #Optimize delta to minimize the disparity + delta 

# device = 'cpu'
# dataset_path="models/Dataset_20.pickle"
# with open(dataset_path, "rb") as f:
#     rec_dataset = pickle.load(f)
# model_path="models/model_20.model"
# model = BaseRecModel(rec_dataset.feature_num, rec_dataset).to(device)
# model.load_state_dict(torch.load(model_path))
# test_data = rec_dataset.test_data
# item_feature_matrix = rec_dataset.item_feature_matrix
# user_feature_matrix = rec_dataset.user_feature_matrix
# CF_disp, _ = get

# #optimize for delta
# ld = 0 #lambda
# learning_rate = 0.01 #not sure
# delta = 0 #intial unknown

# #forward
# CF_disp**2 + ld*delta

# #backpropagation with derivative wrt delta 
# # = gradient descent with adam optimizer

# #update param delta 


def explainability_scores(self, feature):
#generate feature-based explanations based on updated exposures 
    og_exp0 = self.og_exposure["G0"]
    og_exp1 = self.og_exposure["G1"]
    
    cf_exp0 = self.exposure["G0"]
    cf_exp1 = self.exposure["G1"]

    m = len(user_feature_matrix[0])     #number of users
    K = 20                              #length of recommendation lists
    beta = 0.5

    validity = ((og_exp0 - og_exp1) / m * K)  - ((cf_exp0 - cf_exp1) / m * K)
    proximity = (np.linalg.norm(delta) **2)
    ES = validity - beta * proximity


    return(ES)
