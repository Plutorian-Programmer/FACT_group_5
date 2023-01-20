
# We have two groups of users, g0 and g1, and we want to recommend items to them.
# exposure of recommendation model g is the number of items in R that g0 has interacted with.

def exposure(g, R, user_history):
    g_exposure = 0
    for user in R:
        if user_history[user][0] == g:
            g_exposure += 1
    return g_exposure

# calculate a quantification measure for diparity
def disparity(R, user_history):
    g0_exposure = exposure(0, R, user_history)
    g1_exposure = exposure(1, R, user_history)

    # take the difference between the two sides of the equalities as a quantification measure for disparity
    # theta_DP = abs(G1) * Exposure (G0 |R𝐾) − abs(G0) * Exposure (G1 |R𝐾)
    ##### weet niet hoe ik de absolute waarde van g0 moet nemen, is dat de totale lengte van de lijst?
    theta_DP = abs(g1_exposure) * g0_exposure - abs(g0_exposure) * g1_exposure
    return theta_DP

#Counterfactual reasoning
