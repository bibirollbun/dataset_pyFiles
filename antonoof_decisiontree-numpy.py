import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()


X = iris.data
y = iris.target


print(f'features = {X[0]}, target = {y[0]}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)


predictions_sklearn = tree.predict(X_test)

accuracy = accuracy_score(y_test, predictions_sklearn)
print(f"Accuracy: {accuracy}")


class TreeNode:
    def __init__(self, feature=None, threshold=None, val=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val

    def __str__(self):
        left_str = str(self.left) if self.left else "None"
        right_str = str(self.right) if self.right else "None"

        return f"TreeNode(val={self.val}, left={left_str}, right={right_str})"


node = TreeNode()

node.val = 1
node.left = TreeNode(val=2, left=3, right=4)

print(node)


class MyDecisionTreeClassifier:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.Tree = None

    def gini(self, y):
        """Calculate the Gini impurity for a list of classes."""
        total = len(y)

        if total == 0:
            return 0

        counts = Counter(y)
        impurity = 1 - sum((count / total) ** 2 for count in counts.values())

        return impurity

    def best_split(self, X, y):
        """Find the best feature and threshold to split the data."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if np.any(left_indices) and np.any(right_indices):
                    left_classes = y[left_indices]
                    right_classes = y[right_indices]

                    gini_left = self.gini(left_classes)
                    gini_right = self.gini(right_classes)
                    weighted_gini = (len(left_classes) * gini_left + len(right_classes) * gini_right) / len(y)

                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y):
        """Recursively build the decision tree."""
        if len(set(y)) == 1:
            return TreeNode(val=y[0])

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return TreeNode(val=Counter(y).most_common(1)[0][0])

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        # Attention❗ I'm not using the best partitioning for 1 feature,
        # I'm going into recursion, which will cause the trees to grow down to max Gini
        left_node = self.build_tree(X[left_indices], y[left_indices])
        right_node = self.build_tree(X[right_indices], y[right_indices])

        return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self):
        """Fit the decision tree to the data."""
        self.Tree = self.build_tree(self.data, self.target)

    def predict_one(self, node, x):
        """Predict the class for a single sample."""
        if node.val is not None:
            return node.val

        if x[node.feature] < node.threshold:
            return self.predict_one(node.left, x)
        else:
            return self.predict_one(node.right, x)

    def predict(self, X):
        """Predict the classes for a set of samples."""
        return np.array([self.predict_one(self.Tree, x) for x in X])


my_tree = MyDecisionTreeClassifier(X_train, y_train)
my_tree.fit()


predictions = my_tree.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


for y, y_sk, y_our in list(zip(y_test, predictions_sklearn, predictions))[:10]:
    print(f'real = {y}, pred_sklearn = {y_sk}, y_our_model = {y_our}')

