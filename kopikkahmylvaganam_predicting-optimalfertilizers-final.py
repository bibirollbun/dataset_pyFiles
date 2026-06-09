import pandas as pd
import numpy as np
import joblib


# Set random seed for reproducibility
np.random.seed(42)

# Load train.csv
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    print("Train Dataset Loaded Successfully:")
    print(train_df.head())
except FileNotFoundError:
    print("Error: train.csv not found. Please check the file path in the Kaggle Data tab.")

# Load test.csv
try:
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    print("\nTest Dataset Loaded Successfully:")
    print(test_df.head())
except FileNotFoundError:
    print("Error: test.csv not found. Please check the file path in the Kaggle Data tab.")

# Create mappings for categorical variables
soil_types = train_df['Soil Type'].unique()
crop_types = train_df['Crop Type'].unique()
fertilizers = train_df['Fertilizer Name'].unique()
soil_type_map = {soil: idx for idx, soil in enumerate(soil_types)}
crop_type_map = {crop: idx for idx, crop in enumerate(crop_types)}
fertilizer_map = {fert: idx for idx, fert in enumerate(fertilizers)}
reverse_fertilizer_map = {idx: fert for fert, idx in fertilizer_map.items()}

# Encode categorical variables in train.csv
train_df['Soil Type'] = train_df['Soil Type'].map(soil_type_map)
train_df['Crop Type'] = train_df['Crop Type'].map(crop_type_map)
train_df['Fertilizer Name'] = train_df['Fertilizer Name'].map(fertilizer_map)

# Encode categorical variables in test.csv, mapping unseen categories to 0
test_df['Soil Type'] = test_df['Soil Type'].map(soil_type_map).fillna(0).astype(int)
test_df['Crop Type'] = test_df['Crop Type'].map(crop_type_map).fillna(0).astype(int)

# Extract features and target
X_train_full = train_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']].values
y_train_full = train_df['Fertilizer Name'].values
X_test = test_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']].values
test_ids = test_df['id'].values

# Normalize numerical features
def normalize(X, means=None, stds=None):
    X_norm = X.copy().astype(float)
    indices = [0, 1, 2, 5, 6, 7]  # Numerical features: Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous
    if means is None or stds is None:
        means = np.mean(X[:, indices], axis=0)
        stds = np.std(X[:, indices], axis=0) + 1e-8
    for i, idx in enumerate(indices):
        X_norm[:, idx] = (X_norm[:, idx] - means[i]) / stds[i]
    return X_norm, means, stds

# Normalize training and test data
X_train_full, means, stds = normalize(X_train_full)
X_test, _, _ = normalize(X_test, means, stds)

# Split training data into train and validation sets
indices = np.random.permutation(len(X_train_full))
train_size = int(0.8 * len(X_train_full))
train_idx, val_idx = indices[:train_size], indices[train_size:]
X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

# Verify preprocessing
print("\nTrain Data Preprocessed Successfully:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("\nTest Data Preprocessed Successfully:")
print("X_test shape:", X_test.shape)
print("Sample test_ids:", test_ids[:5])
print("\nClass Distribution in y_train:")
class_counts = pd.Series(y_train).value_counts()
for idx, count in class_counts.items():
    print(f"Fertilizer {reverse_fertilizer_map[idx]}: {count} samples")
print("\nSoil Type Map:", soil_type_map)
print("Crop Type Map:", crop_type_map)




# Compute class weights for imbalance
class_counts = np.bincount(y_train, minlength=len(fertilizers))
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / np.sum(class_weights) * len(fertilizers)

class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_counts = class_counts

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
        self.n_classes = None

    def entropy(self, y, weights=None):
        if len(y) == 0:
            return 0
        if weights is None:
            weights = np.ones(len(y))
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros(self.n_classes)
        for i in range(len(y)):
            weighted_counts[y[i]] += weights[i]
        probs = weighted_counts / (np.sum(weighted_counts) + 1e-8)
        return -np.sum(probs * np.log2(probs + 1e-8))

    def information_gain(self, X, y, feature_idx, threshold, weights=None):
        if weights is None:
            weights = np.ones(len(y))
        parent_entropy = self.entropy(y, weights)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        n_left, n_right = np.sum(weights[left_mask]), np.sum(weights[right_mask])
        if n_left < 1e-8 or n_right < 1e-8:
            return 0
        left_entropy = self.entropy(y[left_mask], weights[left_mask])
        right_entropy = self.entropy(y[right_mask], weights[right_mask])
        n = np.sum(weights)
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def find_best_split(self, X, y, weights=None):
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left < 1 or n_right < 1:
                    continue
                gain = self.information_gain(X, y, feature_idx, threshold, weights)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0, weights=None):
        if len(X) == 0 or len(y) == 0:
            return DecisionTreeNode(value=0, class_counts=np.zeros(self.n_classes))
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            class_counts = np.bincount(y, minlength=self.n_classes) if len(y) > 0 else np.zeros(self.n_classes)
            return DecisionTreeNode(value=np.argmax(class_counts) if len(y) > 0 else 0, class_counts=class_counts)
        feature_idx, threshold = self.find_best_split(X, y, weights)
        if feature_idx is None:
            class_counts = np.bincount(y, minlength=self.n_classes) if len(y) > 0 else np.zeros(self.n_classes)
            return DecisionTreeNode(value=np.argmax(class_counts) if len(y) > 0 else 0, class_counts=class_counts)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            class_counts = np.bincount(y, minlength=self.n_classes) if len(y) > 0 else np.zeros(self.n_classes)
            return DecisionTreeNode(value=np.argmax(class_counts) if len(y) > 0 else 0, class_counts=class_counts)
        left_weights = weights[left_mask] if weights is not None else None
        right_weights = weights[right_mask] if weights is not None else None
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1, left_weights)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1, right_weights)
        return DecisionTreeNode(feature_idx, threshold, left, right)

    def fit(self, X, y, weights=None):
        self.n_classes = len(np.unique(y))
        self.root = self.build_tree(X, y, weights=weights)

    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_single(x, node.left)
        return self.predict_single(x, node.right)

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])

    def predict_proba_single(self, x, node):
        if node.class_counts is not None:
            total = max(np.sum(node.class_counts), 1e-8)
            return node.class_counts / total
        if x[node.feature_idx] <= node.threshold:
            return self.predict_proba_single(x, node.left)
        return self.predict_proba_single(x, node.right)

    def predict_proba(self, X):
        return np.array([self.predict_proba_single(x, self.root) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_weights = class_weights[y[indices]]
        return X[indices], y[indices], sample_weights

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample, sample_weights = self.bootstrap_sample(X, y)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_sample[:, feature_indices]
            tree.fit(X_subset, y_sample, weights=sample_weights)
            self.trees.append((tree, feature_indices))

    def predict_proba(self, X):
        n_classes = len(fertilizers)
        proba_sum = np.zeros((X.shape[0], n_classes))
        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            proba = tree.predict_proba(X_subset)
            proba_sum += proba
        return proba_sum / self.n_trees

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# Test DecisionTree independently with class weighting
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train, weights=class_weights[y_train])
y_val_pred = dt.predict(X_val)
y_val_proba = dt.predict_proba(X_val)
print("Decision Tree predictions on validation set:", y_val_pred[:5])
print("Decision Tree top-3 predicted fertilizers:")
for i in range(5):
    top3_indices = np.argsort(y_val_proba[i])[::-1][:3]
    top3_fertilizers = [reverse_fertilizer_map[idx] for idx in top3_indices]
    print(f"Sample {i}: {top3_fertilizers}")


# Define MAP@3 evaluation metric
def map_at_3(y_true, y_pred_proba):
    n = len(y_true)
    score = 0.0
    for i in range(n):
        top3_indices = np.argsort(y_pred_proba[i])[::-1][:3]
        relevant = 0
        for k in range(3):
            if top3_indices[k] == y_true[i]:
                relevant = 1
                score += relevant / (k + 1)
                break
    return score / n

# Train Random Forest with class weighting
rf = RandomForest(n_trees=20, max_depth=7)
rf.fit(X_train, y_train)

# Evaluate on validation set
y_val_proba = rf.predict_proba(X_val)
map_score = map_at_3(y_val, y_val_proba)
print(f"Random Forest MAP@3 on Validation Set (Weighted): {map_score:.4f}")

# Predict top 3 fertilizers for test data
y_test_proba = rf.predict_proba(X_test)
top3_preds = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
top3_fertilizers = [[reverse_fertilizer_map[idx] for idx in pred] for pred in top3_preds]

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(preds) for preds in top3_fertilizers]
})

# Save submission file
submission.to_csv('/kaggle/working/submission_weighted.csv', index=False)
print("Submission file created: /kaggle/working/submission_weighted.csv")

# Verify submission
submission_check = pd.read_csv('/kaggle/working/submission_weighted.csv')
print(submission_check.head())


# Save the trained RandomForest model
joblib.dump(rf, '/kaggle/working/random_forest_model.joblib')
print("Model saved to /kaggle/working/random_forest_model.joblib")



# Load the model
loaded_rf = joblib.load('/kaggle/working/random_forest_model.joblib')

# Use it like before
y_test_proba_loaded = loaded_rf.predict_proba(X_test)
top3_preds_loaded = np.argsort(y_test_proba_loaded, axis=1)[:, -3:][:, ::-1]


