#STRUCTURE


######INPUT
# "m" users in U
# "n" items in V
# "f" mentioned feature in F; "card", "battery" etc. 
# "s" sentiment in S; s = +- 1.
# 'r' size of unique f (features) in R.

# T (Interaction set); for which we have a set: {(u,v) given that u in U, and v in V}. NOTE: elements from T are 2 dimensional.
# W (Review information); for which we have a set: {(u_l,v_l,f_l,s_l)} with l in 1, ..., N.
# M (rating scale); 1, ..., 5 (rating in stars). 

####### CONSTRUCTION
### part 1 
#Af (vector van user-feature); a mxr matrix
#delta (vector delta u) is parameter die later bij de user-feature vector wordt gevoegd, hiervan wil je de optimale waarde leren
""" question, what is starting value for delta """

#Bf (vector van item-feature); a nxr matrix
#delta (vector delta v) same same

### part 2

# forwarding these vectors in the general recommendation model (g) (use elements-wise product merge), will output the counterfactual result Rk(cf)

##  R𝐾 = {R(𝑢1,𝐾),R(𝑢2,𝐾),··· ,R(𝑢𝑚,𝐾)} containing all users’ top-𝐾 recommendation lists

# now each item has a different recommendation distribution than before, so we can calculate the disparity again
#disparity between G0 (most popular 20% items) and G1 (least popular 80% items) caluclation adhv formule (10)
# long-tail rate = verhouding pop items 20% tov 80% minst pop items = hun manier van groep verdeling

### part 3 

# disparity formula

# disparity + delta

#loop to update (minimize) delta (backpropagation / delta)
#aka, minimize the change for a maximal disparity of fairness 
