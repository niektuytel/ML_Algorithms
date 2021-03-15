import numpy as np
import matplotlib.pyplot as plt


# see info ./images/cost.png & ./cost_functions/cost_function.py & ./cost_functions/README.md
def calc_cost(theta, x, y):
    m = len(y)
    prediction = x.dot(theta)

    cost = (1/2 * m) * np.sum((prediction - y) ** 2)
    return cost

def stochastic_gradient_descent(x, y, theta, learning_rate=0.01, iterations=10):
    '''
    x = Matrix of x with added bias units
    y = Vector of y
    theta = Vector of thetas np.random.randn(j, 1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''

    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = x[rand_ind,:].reshape(1, x.shape[1])
            Y_i = y[rand_ind].reshape(1, 1)
            prediction = np.dot(X_i, theta)

            theta = theta -(1/m) * learning_rate * (X_i.T.dot((prediction - Y_i)))
            cost += calc_cost(theta, X_i, Y_i)

        cost_history[it] = cost

    return theta, cost_history

def sample():
    X = 2 * np.random.rand(100,1)
    X_b = np.c_[np.ones((len(X),1)),X]
    Y = 7 * X + np.random.randn(100,1)

    learning_rate = 0.01
    iteration_amount = 1000
    theta = np.random.randn(2,1)

    theta, cost_history = stocashtic_gradient_descent(X_b, Y, theta, learning_rate, iteration_amount)
    fig,ax = plt.subplots(figsize=(10,8))

    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')
    theta = np.random.randn(2,1)

    _ = ax.plot(
        range(iteration_amount),
        cost_history,
        'b.'
    )
    plt.show()


if	__name__ == '__main__':
    sample()