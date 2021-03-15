import numpy as np
import matplotlib.pyplot as plt

# see info ./images/cost.png & ./cost_functions/cost_function.py & ./cost_functions/README.md
def calc_cost(theta, x, y):
    m = len(y)
    prediction = x.dot(theta)

    cost = (1/2 * m) * np.sum((prediction - y) ** 2)
    return cost

def batch_gradient_descent(x, y, theta, learning_rate=0.01, iterations=100):
    '''
    x = Matrix of X with addded bias units
    y = Vector of y
    theta = Vector of thetas np.random.randn(j, 1) 
    learning_rate
    iterations = no of interations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    # total amount of values
    m = len(y)
    
    # empty array
    cost_history = np.zeros(iterations)
    
    # filled array
    filled_arr = np.ones( len(x) )

    # merge batch_X to filled_arr
    batch_X = np.c_[filled_arr, x]

    for it in range(iterations):
        prediction = np.dot(batch_X, theta)
        new_prediction = (prediction - y)

        # print(x, new_prediction)
        # print("\n\n\n")
        # return
        theta = theta - (1/m) * learning_rate * ( x.T.dot( new_prediction ) )
        cost_history[it] = calc_cost(theta, batch_X, y)

    return theta, cost_history

def sample():
    # generate random values
    X = 2 * np.random.rand(100,1)
    Y = 4 + 3 * X + np.random.randn(100,1)
    theta = np.random.randn(2,1)

    # interaction settings
    learning_rate = 0.01
    iterations_amount = 1000

    X_b = np.c_[ np.ones((len(X),1)),X ]

    # I
    theta, cost_history = batch_gradient_descent(X,Y,theta,learning_rate,iterations_amount)
    _, ax = plt.subplots(figsize=(10,8))

    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')

    _ = ax.plot( range(iterations_amount), cost_history, 'b.' )
    plt.show()

if __name__ == "__main__":
    sample()