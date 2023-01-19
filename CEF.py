#STRUCTURE


#Af (vector van user-feature)
#delta (vector delta u) is parameter die later bij de user-feature vector wordt gevoegd, hiervan wil je de optimale waarde leren
""" question, what is starting value for delta """

#Bf (vector van item-feature)
#delta (vector delta v) same same

# forwarding these vectors in the general recommendation model (g), will output the counterfactual result Rk(cf)

##  R𝐾 = {R(𝑢1,𝐾),R(𝑢2,𝐾),··· ,R(𝑢𝑚,𝐾)} containing all users’ top-𝐾 recommendation lists

# now each item has a different recommendation distribution than before, so we can calculate the disparity again
#disparity between G0 (most popular 20% items) and G1 (least popular 20% items ) caluclation adhv formule (10)
# long-tail rate = verhouding pop items 20% tov 80% minst pop items = hun manier van groep verdeling

# disparity + delta

#loop to update (minimize) delta (backpropagation / delta)
#aka, minimize the change for a maximal disparity of fairness 