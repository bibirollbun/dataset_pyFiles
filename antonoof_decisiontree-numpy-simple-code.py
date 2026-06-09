import numpy as np
import pandas as pd

from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("/kaggle/input/clear-data/train.csv")
test = pd.read_csv("/kaggle/input/clear-data/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


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

        # Attentionâ�— I'm not using the best partitioning for 1 feature,
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


X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

my_tree = MyDecisionTreeClassifier(X_train_np, y_train_np)
my_tree.fit()


X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

predictions = my_tree.predict(X_test_np)

accuracy = accuracy_score(y_test_np, predictions)
print(f"Accuracy: {accuracy}")


test_preds = my_tree.predict(test.drop(columns=["id"]).to_numpy())
final_preds = (test_preds > 0.5).astype(int)


submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()

