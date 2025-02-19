from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
import numpy as np
from scipy.optimize import minimize


def misclassification_error(y):
    errors = sum(1 for label in y if label != y.mode()[0])  # Count misclassified instances
    return 0 if len(y) == 0 else errors / len(y)


def best_split(df, feature, target, loss_function):
    """finds best split for the dataset

    Parameters:
    - df: DataFrame containing the data
    - target: Column name of the target variable
    - feature: the feature to split on
    - loss_function: loss used to evaluate splits

    Returns:
    - best split (threshold) and associated loss
    """

    # Extract unique thresholds
    thresholds = np.unique(df[feature].values)
    best_threshold = None
    best_loss = float("inf")


    # Iterate over all possible thresholds
    for threshold in thresholds:
        left_mask = df[feature] <= threshold
        right_mask = ~left_mask

        left_loss = loss_function(df[left_mask][target])
        right_loss = loss_function(df[right_mask][target])

        # Compute weighted loss
        weighted_loss = (left_loss * sum(left_mask) + right_loss * sum(right_mask))

        # Update best split if this one is better
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_threshold = threshold

    return best_threshold, best_loss

def entropy(y):

    label_counts = y.value_counts().to_dict()
    probabilities = [count / len(y) for count in label_counts.values()]

    entropy_value = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return 0 if len(y)==0 else entropy_value


def recursive_best_split(df, target, features, depth=0, max_depth=3, loss_function=entropy):
    """
    Recursively finds the best split for the dataset to create a decision tree.

    Parameters:
    - df: DataFrame containing the data
    - target: Column name of the target variable
    - features: List of feature column names
    - depth: Current depth of recursion
    - max_depth: Maximum depth allowed for splitting
    - loss_function: loss used to evaluate splits


    Returns:
    - A dictionary representing the decision tree
    """

    # Base case: Stop if max depth is reached or all labels are the same
    if depth >= max_depth or len(set(df[target])) == 1:
        return df[target].mode()[0]  # Return most common class

    best_feature, best_threshold, best_loss = None, None, float("inf")

    # Find the best split among all features
    for feature in features:
        threshold, loss = best_split(df, feature, target, loss_function)
        if loss < best_loss:
            best_loss = loss
            best_feature = feature
            best_threshold = threshold

    # If no valid split is found, return the most common class
    if best_feature is None:
        return df[target].mode()[0]

    # Recursively split the left and right child nodes
    left_tree = recursive_best_split(df[df[best_feature] <= best_threshold], target, features, depth + 1, max_depth, loss_function)
    right_tree = recursive_best_split(df[df[best_feature] > best_threshold], target, features, depth + 1, max_depth, loss_function)

    # Return the decision tree structure
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree,
    }

def predict_tree(tree, values):
    """Recursively traverse the decision tree to get predictions."""
    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    threshold = tree["threshold"]

    if values[feature] <= threshold:
        return predict_tree(tree["left"], values)
    else:
        return predict_tree(tree["right"], values)

###FROM LECTURE

def best_split(df, feature, target, loss_function):
    """finds best split for the dataset
    
    Parameters:
    - df: DataFrame containing the data
    - target: Column name of the target variable
    - feature: the feature to split on
    - loss_function: loss used to evaluate splits
    
    Returns:
    - best split (threshold) and associated loss
    """

    # Extract unique thresholds
    thresholds = np.unique(df[feature].values)
    best_threshold = None
    best_loss = float("inf")
    

    # Iterate over all possible thresholds
    for threshold in thresholds:
        left_mask = df[feature] <= threshold
        right_mask = ~left_mask

        left_loss = loss_function(df[left_mask][target])
        right_loss = loss_function(df[right_mask][target])

        # Compute weighted loss
        weighted_loss = (left_loss * sum(left_mask) + right_loss * sum(right_mask))

        # Update best split if this one is better
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_threshold = threshold

    return best_threshold, best_loss


def recursive_best_split(df, target, features, depth=0, max_depth=3, loss_function=entropy):
    """
    Recursively finds the best split for the dataset to create a decision tree.
    
    Parameters:
    - df: DataFrame containing the data
    - target: Column name of the target variable
    - features: List of feature column names
    - depth: Current depth of recursion
    - max_depth: Maximum depth allowed for splitting
    - loss_function: loss used to evaluate splits

    
    Returns:
    - A dictionary representing the decision tree
    """

    # Base case: Stop if max depth is reached or all labels are the same
    if depth >= max_depth or len(set(df[target])) == 1:
        return df[target].mode()[0]  # Return most common class

    best_feature, best_threshold, best_loss = None, None, float("inf")

    # Find the best split among all features
    for feature in features:
        threshold, loss = best_split(df, feature, target, loss_function)
        if loss < best_loss:
            best_loss = loss
            best_feature = feature
            best_threshold = threshold

    # If no valid split is found, return the most common class
    if best_feature is None:
        return df[target].mode()[0]

    # Recursively split the left and right child nodes
    left_tree = recursive_best_split(df[df[best_feature] <= best_threshold], target, features, depth + 1, max_depth, loss_function)
    right_tree = recursive_best_split(df[df[best_feature] > best_threshold], target, features, depth + 1, max_depth, loss_function)

    # Return the decision tree structure
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree,
    }

def predict_tree(tree, values):
    """Recursively traverse the decision tree to get predictions."""
    if not isinstance(tree, dict):
        return tree  

    feature = tree["feature"]
    threshold = tree["threshold"]
    
    if values[feature] <= threshold:
        return predict_tree(tree["left"], values)
    else:
        return predict_tree(tree["right"], values)

def plot_decision_boundaries(decision_tree, df, features, target, reg=False):
    """
    Plots the decision boundaries of a trained decision tree using the updated standalone predict function.
    
    Parameters:
    - decision_tree: The decision tree dictionary built using recursive_best_split
    - df: DataFrame containing the data
    - features: List of feature column names (should be exactly 2 for visualization)
    - target: Column name of the target variable
    """
    # Encode target labels into numeric values
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df[target]))}
    inverse_class_mapping = {idx: label for label, idx in class_mapping.items()}
    colors = {list(class_mapping.items())[0][0]: "steelblue", list(class_mapping.items())[1][0]: "coral"}

    # Generate a grid of points for visualization
    x_min, x_max = df[features[0]].min() - 1, df[features[0]].max() + 1
    y_min, y_max = df[features[1]].min() - 1, df[features[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Make predictions for all grid points and encode categorical values to numeric labels
    Z = np.array([class_mapping[predict_tree(decision_tree, {features[0]: x, features[1]: y})] 
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis", linewidths=0)
    plt.contour(xx, yy, Z, colors="black", linewidths=.5)  # Black boundary lines


    # Scatter plot of original data points with colors corresponding to classes
    if(not reg):
      for class_label, class_idx in class_mapping.items():
        subset = df[df[target] == class_label]
        plt.scatter(subset[features[0]], subset[features[1]], label=class_label, color=colors[class_label])
    else:
      scatter = plt.scatter(df[features[0]], df[features[1]], c=df[target], cmap="viridis", edgecolors=None, alpha=.6, s=20)
      plt.colorbar(scatter, label=target)


    # Labels and legend
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Decision Boundaries of the Decision Tree")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.show()
