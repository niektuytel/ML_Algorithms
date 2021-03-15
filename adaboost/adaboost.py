# ##########################################################################################################
# ##                                       Adaboost from scratch                                          ## 
# ##########################################################################################################
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import seaborn as sns
sns.set_style("white")


# Toy Dataset
x1 = np.array([.1,.2,.4,.8, .8, .05,.08,.12,.33,.55,.66,.77,.88,.2,.3,.4,.5,.6,.25,.3,.5,.7,.6])
x2 = np.array([.2,.65,.7,.6, .3,.1,.4,.66,.77,.65,.68,.55,.44,.1,.3,.4,.3,.15,.15,.5,.55,.2,.4])
y  = np.array([1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1])
X  = np.vstack((x1,x2)).T

def AdaBoost(X, y, M=20, learning_rate = 1):
    # Initialize of utility variables
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []

    # Initialize the sample weights
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())

    skip = False

    # For m = 1 to M
    for m in range(M):
        # Fit a classifier
        # explain in https://github.com/niektuytel/decision_tree
        estimator = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
        estimator.fit(X, y, sample_weight = sample_weight)
        y_predict = estimator.predict(X)

        # Misclassifications
        incorrect = (y_predict != y)

        # Estimator error
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Boost estimator weights
        estimator_weight = learning_rate * np.log((1. - estimator_error) / estimator_error)

        # Boost sample weights
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Save iteration values
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert to np array for convenience
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Predictions
    preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    print("Accuracy = ", (preds == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list

def plot_AdaBoost(estimators, estimator_weights, X, y, N = 10, ax = None):
    def classify_AdaBoost(x_temp, est, est_weights):
        """ Return classification prediction for given point X and a previously fitted AdaBoost """
        temp_pred = np.asarray([(e.predict(x_temp)).T* w for e, w in zip(est, est_weights)]) / est_weights.sum()
        return np.sign(temp_pred.sum(axis = 0))
    
    """Utility function to plot decision boundary and scatter plot of data"""
    x_min, x_max = X[:, 0].min() - 0.10, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.10, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    zz = np.array(
        [
            classify_AdaBoost(np.array([xi, yi]).reshape(1, -1), estimators, estimator_weights) 
            for xi, yi in zip(np.ravel(xx), np.ravel(yy))
        ]
    )

    # rehape result and plot
    Z = zz.reshape(xx.shape)
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    if ax is None:
        ax = plt.gca()

    ax.contourf(xx, yy, Z, 2, cmap="RdBu", alpha=0.5)
    ax.contour(xx, yy, Z, 2, cmap="RdBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")

def sample():
    fig = plt.figure()
    values = [1,3,10, 15, 7]
    
    for k,m in enumerate(values):
        fig.add_subplot(2,3,1+k)
        estimator_list, estimator_weight_list, sample_weight_list = AdaBoost(X,y, M=m, learning_rate = 1)
        plot_AdaBoost(estimator_list,estimator_weight_list, X, y, N = 50,ax = None )
        plt.title('Adaboost boundary: M = {},  L = 1'.format(m))
        print(estimator_weight_list)

    plt.show()

# # ##########################################################################################################
# # ##                                       Adaboost with sklearn                                          ## 
# # ##########################################################################################################
# #Import the Libraries
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# def sample():
#     #Import the Dataset
#     Dataset = pd.read_csv("../_DATA/iris_dataset.csv")
#     X = Dataset.iloc[:,:-1].values
#     y = Dataset.iloc[:, -1].values

#     #Splitting the data into train and test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=32)

#     #Adaboost classifier
#     classifier2 = AdaBoostClassifier(random_state=0)
#     classifier2.fit(X_train, y_train)

#     #predict our test set
#     y_pred2 = classifier2.predict(X_test)

#     #evaluate our test set
#     acc_test2 = accuracy_score(y_test, y_pred2)
#     f1_test2 = f1_score(y_test, y_pred2, average= 'weighted')

#     print("Test set results")
#     print("ACCURACY for test set(Adaboost classifier method)",acc_test2)
#     print("F1 SCORE for test set(Adaboost classifier method)",f1_test2)








if __name__ == "__main__":
    sample()