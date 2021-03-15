# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader


# Select Most common output value in a list of rows.
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Calculate the Gini index for a split dataset
def gini_index(groups, classes = [0, 1]):
    # count all samples at split point
    n_instances = float(
        sum([len(group) for group in groups])
    )

    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))

        # avoid devide by zero
        if size == 0:
            continue

        # score the group based on the score for each class
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    
    return gini

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    
    return {"index":b_index, "value":b_value, "groups":b_groups}

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del(node["groups"])

    # check for a no split 
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return 

    # process left child 
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth + 1)

    # process right child 
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth + 1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Sample, Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]

# -----------------------------------------------------------

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accurancy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Classification and regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()

    # # testing
    # print_tree(tree)

    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)

    return (predictions)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = decision_tree(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# ###########################################################
# ##                Basic usage of tree                    ##
# ###########################################################
# dataset = [
#     [2.771244718,1.784783929,0],
#     [1.728571309,1.169761413,0],
#     [3.678319846,2.81281357,0],
#     [3.961043357,2.61995032,0],
#     [2.999208922,2.209014212,0],
#     [7.497545867,3.162953546,1],
#     [9.00220326,3.339047188,1],
#     [7.444542326,0.476683375,1],
#     [10.12493903,3.234550982,1],
#     [6.642287351,3.319983761,1]
# ]

# # With a maximum depth of 1 (the second parameter in the call to the build_tree() function), 
# # we can see that the tree uses the perfect split we discovered in the previous section. 
# # This is a tree with one node, also called a decision stump.
# tree = build_tree(dataset, 1, 1)

# # Increasing the maximum depth to 2, we are forcing the tree to make splits even when none are required. 
# # The X1 attribute is then used again by both the left and right children of the root node to split up the already perfect mix of classes.
# tree = build_tree(dataset, 2, 1)

# # force one more level of splits with a maximum depth of 3.
# # These tests show that there is great opportunity to refine the implementation to avoid unnecessary splits. This is left as an extension.
# # Now that we can create a decision tree, let’s see how we can use it to make predictions on new data.
# tree = build_tree(dataset, 3, 1)

# print_tree(tree)

# ###################################################################

# # dataset = [[2.771244718,1.784783929,0],
# # 	[1.728571309,1.169761413,0],
# # 	[3.678319846,2.81281357,0],
# # 	[3.961043357,2.61995032,0],
# # 	[2.999208922,2.209014212,0],
# # 	[7.497545867,3.162953546,1],
# # 	[9.00220326,3.339047188,1],
# # 	[7.444542326,0.476683375,1],
# # 	[10.12493903,3.234550982,1],
# # 	[6.642287351,3.319983761,1]]
#  
# # #  predict with a stump
# # stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# # for row in dataset:
# # 	prediction = predict(stump, row)
# # 	print('Expected=%d, Got=%d' % (row[-1], prediction))




###########################################################
##                     Sample usage                      ## 
###########################################################

# Test CART on Bank Note dataset
seed(1)

# load and prepare data
filename = "../_DATA/data_banknote_authentication.csv"
dataset = load_csv(filename)

# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, n_folds, max_depth, min_size)

print("Scores: %s" % scores)
print("Mean accurancy: %.3f%%" % (sum(scores) / float(len(scores))))


