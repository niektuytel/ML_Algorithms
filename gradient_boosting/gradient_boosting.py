# ############################################################################################
# ##                            sample without sklearn                                      ##
# ############################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=2):
        if idxs is None: idxs=np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float("inf")
        self.find_varsplit()

    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float("inf"): return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0.00, 0.00

        for i in range(0, self.n - self.min_leaf-1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1; rhs_cnt -=1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i < self.min_leaf or xi == sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi
        
    @property
    def split_name(self): return self.x.columns[self.var_idx]

    @property
    def split_col(self): return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self): return self.score == float("inf")

    def __repr__(self):
        s = f"n: {self.n}; val:{self.val}"
        if not self.is_leaf:
            s += f"; score:{self.score}; split:{self.split}; var:{self.split_name}"
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)

def std_agg(cnt, s1, s2):
    return math.sqrt(
        (s2/cnt) - (s1/cnt) ** 2
    )

# Gen New Data
def get_data():
    x = np.arange(0, 50)
    x = pd.DataFrame({"x":x})

    # Just random uniform distribution in different range
    y1 = np.random.uniform(10,15,10)
    y2 = np.random.uniform(20,25,10)
    y3 = np.random.uniform(0,5,10)
    y4 = np.random.uniform(30,32,10)
    y5 = np.random.uniform(13,17,10)

    y = np.concatenate((y1, y2, y3, y4, y5))
    y = y[:, None]

    return x, y

# simple sample of way it works
def sample():
    # Displaying data
    x, y = get_data()
    # print(x.shape, y.shape)
    # plt.figure()
    # plt.plot(x, y, "o")
    # plt.title("Scatter plot of x vs. y")
    # plt.xlabel("x")
    # plt.ylabel("y")

    xi = x # initialization of input  (value get updated)
    yi = y # initialization of target (value get updated)
    ei = 0 # initialization of error  (value get updated)
    n_estimators = 30 # iterations amount
    n_rows = len(yi)  # number of rows
    predictions_total = 0 # initial prediction 0

    for iteration in range(n_estimators):
        # generate and split the tree if possible
        tree = DecisionTree(xi, yi)
        tree.find_better_split(0)

        # get values from tree  
        left_idx = np.where(xi <= tree.split)[0]
        matched_idx = np.where(xi == tree.split)[0][0] 
        right_idx = np.where(xi > tree.split)[0]
        
        # generate predictions dataset
        predictions = np.zeros(n_rows)
        np.put(
            predictions, 
            left_idx, 
            np.repeat(np.mean(yi[left_idx]), matched_idx)# replace `left` side mean y
        )
        np.put(
            predictions, 
            right_idx, 
            np.repeat(np.mean(yi[right_idx]), n_rows - matched_idx)# replace `right` side mean y
        )
        
        predictions = predictions[:,None]# make long vector (nx1) in compatible with y
        predictions_total = predictions_total + predictions# final prediction will be previous prediction value + new prediction of residual

        ei = y - predictions_total  # needed originl y here as residual always from original y    
        yi = ei # update yi as residual to reloop
        
        # plotting after prediction
        xa = np.array(x.x) # column name of x is x 
        order = np.argsort(xa)
        xs = np.array(xa)[order]
        ys = np.array(predictions_total)[order]
        
        #epreds = np.array(epred[:,None])[order]

        if iteration % 5 == 0:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13,2.5))

            ax1.plot(x,y, 'o')
            ax1.plot(xs, ys, 'r')
            ax1.set_title(f'Prediction (Iteration {iteration+1})')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y / y_pred')

            ax2.plot(x, ei, 'go')
            ax2.set_title(f'Residuals vs. x (Iteration {iteration+1})')
            ax2.set_xlabel('x')
            ax2.set_ylabel('Residuals')

    plt.show()

if __name__ == "__main__":
    sample()



