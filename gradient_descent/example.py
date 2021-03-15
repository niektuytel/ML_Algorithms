import numpy as np
import matplotlib.pyplot as plt
import batch_gradient_descent
import mini_batch_gradient_descent
import stochastic_gradient_descent 

def batch_sample():
    X = 2 * np.random.rand(100,1)
    Y = 4 +3 * X+np.random.randn(100,1)
    lr = 0.01
    n_iter = 1000
    theta = np.random.randn(2,1)
    X_b = np.c_[np.ones((len(X),1)),X]

    # I
    theta,cost_history = batch_gradient_descent.batch_gradient_descent(X,Y,theta,lr,n_iter)
    _,ax = plt.subplots(figsize=(10,8))

    ax.set_title("batch sample")
    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')

    _=ax.plot(range(n_iter),cost_history,'b.')

def mini_batch_sample():
    X = 2 * np.random.rand(100,1)
    Y = 4 +3 * X+np.random.randn(100,1)
    lr = 0.01
    n_iter = 1000
    theta = np.random.randn(2,1)
    X_b = np.c_[np.ones((len(X),1)),X]

    # III
    theta,cost_history = mini_batch_gradient_descent.mini_batch_gradient_descent(X,Y,theta,lr,n_iter)
    fig,ax = plt.subplots(figsize=(10,8))

    ax.set_title("mini batch sample")
    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')
    theta = np.random.randn(2,1)

    _=ax.plot(range(n_iter),cost_history,'b.')

def stochastic_sample():
    X = 2 * np.random.rand(100,1)
    Y = 4 +3 * X+np.random.randn(100,1)
    lr = 0.01
    n_iter = 1000
    theta = np.random.randn(2,1)
    X_b = np.c_[np.ones((len(X),1)),X]


    # II
    X_b = np.c_[np.ones((len(X),1)),X]
    theta,cost_history = stochastic_gradient_descent.stochastic_gradient_descent(X_b,Y,theta,lr,n_iter)
    fig,ax = plt.subplots(figsize=(10,8))


    ax.set_title("stochastic sample")
    ax.set_ylabel('{J(Theta)}',rotation=0)
    ax.set_xlabel('{Iterations}')
    theta = np.random.randn(2,1)

    _=ax.plot(range(n_iter),cost_history,'b.')

def all_over_sample():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    learning_rates =[(2000,0.001),(500,0.01),(200,0.05),(100,0.1)]
    count = 0

    for iterations, learning_rate in learning_rates:
        X = 2 * np.random.rand(100,1)
        Y = 4 + 3 * X+np.random.randn(100,1)
        theta = np.random.randn(2,1)
        X_b = np.c_[ np.ones( (len(X),1) ),X ]

        theta = np.random.randn(2,1)
        pred_prev = X_b.dot(theta)

        theta, cost_history = batch_gradient_descent.batch_gradient_descent(X,Y,theta, learning_rate, iterations)
        pred = X_b.dot(theta)

        # Display
        count += 1    
        ax = fig.add_subplot(4, 2, count)

        count += 1
        ax1 = fig.add_subplot(4, 2,count)

        ax.set_title("learning_rate:{}".format(learning_rate))
        ax1.set_title("Iterations:{}".format(iterations))
        _ = ax.plot(X,Y,'b.')
        _ = ax1.plot(range(iterations), cost_history, 'b.')
        _ = ax.plot(X, pred_prev, 'r-', alpha = 0.1)
        _ = ax.plot(X, pred,'r-', alpha = 1)

if	__name__ == '__main__':
    batch_sample()
    mini_batch_sample()
    stochastic_sample()

    all_over_sample()
    plt.show()




    