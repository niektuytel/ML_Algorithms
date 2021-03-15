import numpy as np
import matplotlib.pyplot as plt

# see info ./images/cost.png & ./cost_functions/cost_function.py & ./cost_functions/README.md
def calc_cost(theta, x, y):
    m = len(y)
    prediction = x.dot(theta)

    cost = (1/2 * m) * np.sum((prediction - y) ** 2)
    return cost

def mini_batch_gradient_descent(x, y, theta, learning_rate=0.01, iterations=10, batch_size=20):
    '''
    x = Matrix of x without added bias units
    y = Vector of y
    theta = Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''
    # total amount of values
    m = len(y)
    
    # empty array, size = iterations
    cost_history = np.zeros(iterations)
    
    # number of data training samples
    batches_amount = int(m / batch_size)

    for it in range(iterations):
        cost = 0.0

        # get random [x, y] value
        index = np.random.permutation(m)
        x = x[index]
        y = y[index]

        for i in range(0, m, batch_size):
            # values of batch_size
            batch_X = x[i:i + batch_size]
            batch_Y = y[i:i + batch_size]

            # fill array with 1.
            filled_arr = np.ones( len(batch_X) )

            # merge batch_X to filled_arr
            batch_X = np.c_[filled_arr, batch_X]
            prediction = np.dot(batch_X, theta)

            # generate new position
            step = batch_X.T.dot((prediction - batch_Y))
            theta = theta - (1/m) * learning_rate * step
            cost += calc_cost(theta, batch_X, batch_Y)


        cost_history[it] = cost

    return theta, cost_history

def sample():
    # generate random values
    X = 2 * np.random.rand(100,1)
    Y = 4 + 3 * X + np.random.randn(100,1)
    theta = np.random.randn(2,1)

    # interaction settings
    learning_rate = 0.01
    iterations_amount = 1000

    # **CALCULATION**
    theta, cost_history = mini_batch_gradient_descent(X, Y, theta, learning_rate, iterations_amount)

    # display
    fig,ax = plt.subplots(figsize=(10,8))
    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')
    _ = ax.plot(
        range(iterations_amount),
        cost_history, 'b.'
    )
    plt.show()

    # extra
    print("cost history data: \n" + cost_history)


if	__name__ == '__main__':
    sample()
