
import matplotlib.pyplot as plt
import numpy as np

# see ../images/cost.png
def cost_function(theta, x, y): 
    m = len(y)
    prediction = x.dot(theta)

    cost = (1/2 * m) * np.sum((prediction - y)**2)
    return cost

################################################################################
#########################Sample Without numpy ##################################
################################################################################
# # original data set
# X = [1, 2, 3]
# y = [1, 2.5, 3.5]

# # slope of 1 is 0.5
# # slope of 2 is 1.0
# # slope of 3 is 1.5

# hyps = [0.5, 1.0, 1.5] 

# # mutiply the original X values by the theta 
# # to produce hypothesis values for each X
# def multiply_matrix(mat, theta):
#     mutated = []
#     for i in range(len(mat)):
#         mutated.append(mat[i] * theta)

#     return mutated

# # calculate cost by looping each sample
# # subtract hyp(x) from y
# # square the result
# # sum them all together
# def calc_cost(m, X, y):
#     total = 0
#     for i in range(m):
#         squared_error = (y[i] - X[i]) ** 2
#         total += squared_error
    
#     return total * (1 / (2*m))

# # calculate cost for each hypothesis
# for i in range(len(hyps)):
#     hyp_values = multiply_matrix(X, hyps[i])

#     print("Cost for ", hyps[i], " is ", calc_cost(len(X), y, hyp_values))

################################################################################
#########################Sample With numpy #####################################
################################################################################

def sample():
    X = np.array([[1], [2], [3]])
    y = np.array([[1], [2.5], [3.5]])

    get_theta = lambda theta: np.array([[0, theta]])

    thetas = list(map(get_theta, [0.5, 1.0, 1.5]))

    X = np.hstack([np.ones([3, 1]), X])

    for i in range(len(thetas)):
        inner = np.power(((X @ thetas[i].T) - y), 2)
        cost = np.sum(inner) / (2 * len(X))

        print(cost)

if __name__ == "__main__":
    sample()

# Resources
# - https://medium.com/@lachlanmiller_52885/understanding-and-calculating-the-cost-function-for-linear-regression-39b8a3519fcb
# - https://stackoverflow.com/questions/13623113/can-someone-explain-to-me-the-difference-between-a-cost-function-and-the-gradien

