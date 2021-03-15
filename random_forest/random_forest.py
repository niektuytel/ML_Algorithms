# ################################################################################################
# ##                               Random Forest from scratch                                   ##
# ################################################################################################
import random
import csv
import math

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accurancy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

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

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point 
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group 
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue

        score = 0.0
        # score the group based on the score for each class 
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group based on the score for each class 
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = random.randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {"index":b_index, "value": b_value, "groups":b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal 
def split(node, max_depth, min_size, n_features, depth):
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
        node["left"] = get_split(left, n_features)
        split(node["left"], max_depth, min_size, n_features, depth+1)

    # process right child
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right, n_features)
        split(node["right"], max_depth, min_size, n_features, depth+1)

# Build a decicion tree 
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

# Random Forest Algorithm
def random_forests(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

# Split a dataset based on an attribure and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right

# Evaluate an algotithm using a cross validation split
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

        predicted = random_forests(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accurancy = accurancy_metric(actual, predicted)
        scores.append(accurancy)
    return scores


# Test random forest algorithm
random.seed(2)

# load and prepare data
filename = "../_DATA/sonar.all-data.csv"
dataset = load_csv(filename)

# convert string attributes to integer
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

# evaluate Algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(math.sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
    scores = evaluate_algorithm(dataset, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print("Trees: %d" %  n_trees)
    print("Scores: %s" % scores)
    print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))





# ################################################################################################
# ##              A using sample of Random Forest with sklearn, pandas, numpy                   ##
# ################################################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

# import itertools

# RSEED = 50

# def get_data(test_size=0.3, random_state=50):
    
#     # Load in data
#     df = pd.read_csv('https://s3.amazonaws.com/projects-rf/clean_data.csv')

#     # Full dataset: https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system

#     # Extract the labels
#     labels = np.array(df.pop('label'))

#     # 30% examples in test data
#     train, test, train_labels, test_labels = train_test_split(
#         df,
#         labels, 
#         stratify = labels,
#         test_size = test_size, 
#         random_state = random_state
#     )

#     # Imputation of missing values
#     train = train.fillna(train.mean())
#     test = test.fillna(test.mean())

#     return (train, test, train_labels, test_labels)

# def evaluate_model(predictions, probs, train_predictions, train_probs):
#     """Compare machine learning model to baseline performance.
#     Computes statistics and shows ROC curve."""
    
#     baseline = {}
    
#     baseline['recall'] = recall_score(test_labels, 
#                                      [1 for _ in range(len(test_labels))])
#     baseline['precision'] = precision_score(test_labels, 
#                                       [1 for _ in range(len(test_labels))])
#     baseline['roc'] = 0.5
    
#     results = {}
    
#     results['recall'] = recall_score(test_labels, predictions)
#     results['precision'] = precision_score(test_labels, predictions)
#     results['roc'] = roc_auc_score(test_labels, probs)
    
#     train_results = {}
#     train_results['recall'] = recall_score(train_labels, train_predictions)
#     train_results['precision'] = precision_score(train_labels, train_predictions)
#     train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
#     for metric in ['recall', 'precision', 'roc']:
#         print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
#     # Calculate false positive rates and true positive rates
#     base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
#     model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

#     plt.figure(figsize = (8, 6))
#     plt.rcParams['font.size'] = 16
    
#     # Plot both curves
#     plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
#     plt.plot(model_fpr, model_tpr, 'r', label = 'model')
#     plt.legend();
#     plt.xlabel('False Positive Rate'); 
#     plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
#     plt.show();

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Oranges):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     # Plot the confusion matrix
#     plt.figure(figsize = (10, 10))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, size = 24)
#     plt.colorbar(aspect=4)
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, size = 14)
#     plt.yticks(tick_marks, classes, size = 14)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
    
#     # Labeling the plot
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
        
#     plt.grid(None)
#     plt.tight_layout()
#     plt.ylabel('True label', size = 18)
#     plt.xlabel('Predicted label', size = 18)

# # Load in data
# train, test, train_labels, test_labels = get_data(0.3, RSEED)

# # Create the model with 100 trees
# model = RandomForestClassifier(
#     n_estimators    = 100, 
#     random_state    = RSEED, 
#     max_features    = 'sqrt',
#     n_jobs          = -1, 
#     verbose         = 1
# )

# # Features for feature importances
# # features = list(train.columns)

# # Fit on training data
# model.fit(train, train_labels)

# n_nodes = []
# max_depths = []

# # Stats about the trees in random forest
# for index in model.estimators_:
#     n_nodes.append(index.tree_.node_count)
#     max_depths.append(index.tree_.max_depth)

# print(f'Average number of nodes {int(np.mean(n_nodes))}')
# print(f'Average maximum depth {int(np.mean(max_depths))}')

# # Training predictions (to demonstrate overfitting)
# train_rf_predictions = model.predict(train)
# train_rf_probs = model.predict_proba(train)[:, 1]

# # Testing predictions (to determine performance)
# rf_predictions = model.predict(test)
# rf_probs = model.predict_proba(test)[:, 1]

# # Visualization
# evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)