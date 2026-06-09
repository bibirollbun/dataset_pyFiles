import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt, log, pi
from multiprocessing import Pool
from time import time
import joblib
from pathlib import Path
import os




df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
ids = test_df['id']






# --------------------------
# Data Loading and Preprocessing
# --------------------------

def load_data():
    df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    return df

def preprocess_data(df):
    # Fix temperature typo if exists
    if 'Temparature' in df.columns:
        df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    # Drop ID column
    df = df.drop(columns=['id'])
    
    # One-hot encode categorical features
    soil_dummies = pd.get_dummies(df['Soil Type'], prefix='soil', dtype=int)
    crop_dummies = pd.get_dummies(df['Crop Type'], prefix='crop', dtype=int)
    
    # Handle numerical features
    numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    X_numerical = df[numerical_cols].values
    
    # Combine features
    X = np.concatenate([X_numerical, soil_dummies, crop_dummies], axis=1)
    
    # Label encode target
    y, y_labels = pd.factorize(df['Fertilizer Name'])
    print("Label mapping:", dict(enumerate(y_labels)))  # Debug: Show class mapping
    
    return X, y, y_labels

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Custom normalization (only numerical columns, indices 0-5)
    X_train_num = X_train[:, :6]
    X_test_num = X_test[:, :6]
    X_mean = np.mean(X_train_num, axis=0)
    X_std = np.std(X_train_num, axis=0) + 1e-8  # Avoid division by zero
    X_train[:, :6] = (X_train_num - X_mean) / X_std
    X_test[:, :6] = (X_test_num - X_mean) / X_std
    
    return X_train, X_test, y_train, y_test

# --------------------------
# 1. Neural Network - Optimized
# --------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=64, output_size=3, learning_rate=0.001, epochs=500):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.epochs = epochs
        
        # Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        y_onehot = np.eye(self.output_size)[y]
        
        for epoch in range(1, self.epochs + 1):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.softmax(z2)
            
            # Backpropagation
            dz2 = a2 - y_onehot
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.relu_derivative(z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            
            if epoch % 100 == 0:
                loss = -np.mean(np.log(a2[np.arange(len(y)), y] + 1e-9))
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return np.argmax(z2, axis=1)

# --------------------------
# 2. Decision Tree - Optimized
# --------------------------

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds[:10]:  # Limit thresholds evaluated
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if len(y[left_idx]) < self.min_samples_split or len(y[right_idx]) < self.min_samples_split:
                    continue
                
                p = len(y[left_idx]) / len(y)
                gain = self._entropy(y) - p * self._entropy(y[left_idx]) - (1-p) * self._entropy(y[right_idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)
        
        return DecisionTreeNode(feature, threshold, left, right)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

# --------------------------
# 3. Random Forest 
# --------------------------

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=20, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        idxs = np.random.choice(len(y), len(y), replace=True)
        return X[idxs], y[idxs]
    
    def _train_tree(self, args):
        X, y, feat_idxs = args
        tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X[:, feat_idxs], y)
        return (tree, feat_idxs)
    
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features
        
        tree_args = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
            tree_args.append((X_sample, y_sample, feat_idxs))
        
        with Pool() as p:
            self.trees = p.map(self._train_tree, tree_args)
    
    def predict(self, X):
        tree_preds = np.zeros((len(self.trees), X.shape[0]), dtype=int)
        
        for i, (tree, feat_idxs) in enumerate(self.trees):
            tree_preds[i] = tree.predict(X[:, feat_idxs])
        
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

# --------------------------
# Evaluation Functions
# --------------------------

def custom_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = custom_accuracy_score(y_train, y_train_pred)
    test_acc = custom_accuracy_score(y_test, y_test_pred)
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model': model
    }

# --------------------------
# Main Execution 
# --------------------------

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    X, y, y_labels = preprocess_data(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training shape:", X_train.shape, "Test shape:", X_test.shape)
    
    # Initialize models
    models = {
        'Neural Network': NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=128,
            output_size=len(np.unique(y_train)),
            learning_rate=0.001,
            epochs=500
        ),
        'Decision Tree': DecisionTree(max_depth=10),
        'Random Forest': RandomForest(n_trees=50, max_depth=10)
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
        print(f"{name}:")
        print(f"  Train Accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"  Test Accuracy:  {results[name]['test_accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['test_accuracy']:.4f})")

if __name__ == "__main__":
    main()




# --------------------------
# Enhanced Decision Tree Implementation
# --------------------------
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            unique_vals = np.unique(X[:, feature])
            if len(unique_vals) > 10:
                thresholds = np.percentile(X[:, feature], np.linspace(10, 90, 15))
            else:
                thresholds = unique_vals
                
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if (len(y[left_idx]) < self.min_samples_split or 
                    len(y[right_idx]) < self.min_samples_split or
                    len(y[left_idx]) < self.min_samples_leaf or
                    len(y[right_idx]) < self.min_samples_leaf):
                    continue
                
                p = len(y[left_idx]) / len(y)
                gain = self._gini(y) - p * self._gini(y[left_idx]) - (1-p) * self._gini(y[right_idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)
        
        return DecisionTreeNode(feature, threshold, left, right)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

# --------------------------
# Random Forest Implementation
# --------------------------
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=20, 
                 max_features=None, min_samples_leaf=1, n_jobs=4, class_weight=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.trees = []
    
    def _calculate_sample_weights(self, y):
        if self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            n_samples = len(y)
            n_classes = len(class_counts)
            weights = n_samples / (n_classes * class_counts[y] + 1e-8)
            return weights / weights.sum() * n_samples
        return None
    
    def _bootstrap_sample(self, X, y):
        if self.class_weight is not None:
            weights = self._calculate_sample_weights(y)
            idxs = np.random.choice(len(y), len(y), replace=True, p=weights/weights.sum())
        else:
            idxs = np.random.choice(len(y), len(y), replace=True)
        return X[idxs], y[idxs]
    
    def _train_tree(self, args):
        X, y, feat_idxs = args
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        tree.fit(X[:, feat_idxs], y)
        return (tree, feat_idxs)
    
    def fit(self, X, y):
        print(f"\nTraining Random Forest with {self.n_trees} trees (max_depth={self.max_depth})...")
        start_time = time()
        
        self.trees = []
        n_features = X.shape[1]
        
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        elif self.max_features is None:
            self.max_features = n_features
        
        batch_size = min(20, self.n_trees)
        for batch_start in range(0, self.n_trees, batch_size):
            batch_end = min(batch_start + batch_size, self.n_trees)
            print(f"  Training trees {batch_start+1}-{batch_end}...")
            
            tree_args = []
            for i in range(batch_start, batch_end):
                X_sample, y_sample = self._bootstrap_sample(X, y)
                feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
                tree_args.append((X_sample, y_sample, feat_idxs))
            
            with Pool(processes=min(self.n_jobs, batch_size)) as p:
                batch_trees = p.map(self._train_tree, tree_args)
            self.trees.extend(batch_trees)
        
        print(f"Training completed in {time()-start_time:.2f} seconds")
    
    def predict_proba(self, X):
        # Initialize probability array with correct shape
        n_classes = len(np.unique([t.predict(X[:1]) for t, _ in self.trees]))
        proba = np.zeros((X.shape[0], n_classes))
        for tree, feat_idxs in self.trees:
            preds = tree.predict(X[:, feat_idxs])
            for i, pred in enumerate(preds):
                proba[i, pred] += 1
        return proba / len(self.trees)
    
    def predict(self, X):
        batch_size = 1000
        predictions = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            batch_preds = np.zeros((len(self.trees), len(X_batch)), dtype=int)
            for j, (tree, feat_idxs) in enumerate(self.trees):
                batch_preds[j] = tree.predict(X_batch[:, feat_idxs])
            predictions.extend(np.array([Counter(batch_preds[:, k]).most_common(1)[0][0] 
                              for k in range(len(X_batch))]))
        return np.array(predictions)

# --------------------------
# Data Loading and Preprocessing
# --------------------------
def load_and_preprocess():
    print("Loading and preprocessing data...")
    df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    
    if 'Temparature' in df.columns:
        df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    X = df.drop(['id', 'Fertilizer Name'], axis=1)
    y = df['Fertilizer Name'].astype('category').cat.codes
    label_map = df['Fertilizer Name'].astype('category').cat.categories
    
    # Feature engineering
    X['NP_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1e-8)
    X['NK_ratio'] = X['Nitrogen'] / (X['Potassium'] + 1e-8)
    X['Temp_Humidity'] = X['Temperature'] * X['Humidity']
    
    X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
    num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
               'Potassium', 'Phosphorous', 'NP_ratio', 'NK_ratio', 'Temp_Humidity']
    
    # Normalization
    X_num = X[num_cols].values
    X_mean = np.mean(X_num, axis=0)
    X_std = np.std(X_num, axis=0) + 1e-8
    X[num_cols] = (X_num - X_mean) / X_std
    
    return X.values, y.values, label_map

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# --------------------------
# Evaluation Functions
# --------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, label_map):
    print(f"\nTraining {model.__class__.__name__}...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = np.mean(y_train == y_train_pred)
    test_acc = np.mean(y_test == y_test_pred)
    
    # Urea analysis
    urea_idx = np.where(label_map == 'Urea')[0][0]
    urea_test_pred = np.sum(y_test_pred == urea_idx)
    urea_test_actual = np.sum(y_test == urea_idx)
    urea_correct = np.sum((y_test == urea_idx) & (y_test_pred == urea_idx))
    
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'training_time': train_time,
        'model': model
    }

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load data
    X, y, label_map = load_and_preprocess()
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = [
        {
            'name': 'RF3 - Enhanced',
            'params': {
                'n_trees': 150,
                'max_depth': 15,
                'min_samples_split': 8,
                'max_features': 'log2',
                'min_samples_leaf': 2,
                'n_jobs': 4,
                'class_weight': 'balanced'
            }
        }
    ]
    
    # Train and evaluate
    results = {}
    for model in models:
        print("\n" + "="*60)
        print(f"EVALUATING {model['name']}")
        print("="*60)
        
        rf = RandomForest(**model['params'])
        result = evaluate_model(rf, X_train, y_train, X_test, y_test, label_map)
        results[model['name']] = result
        
        print(f"\n{model['name']} Results:")
        print(f"Train Accuracy: {result['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {result['test_accuracy']:.4f}")
        print(f"Training Time:  {result['training_time']:.2f}s")
        print("="*60)
    
    # Show best model
    best_model_name, best_model_results = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print("\n" + "="*50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_model_results['test_accuracy']:.4f}")
    print(f"Training Time: {best_model_results['training_time']:.2f}s")


# --------------------------
# Prediction on Test Set
# --------------------------
if __name__ == "__main__":
    # ... [previous code] ...

    # After selecting best model, add:
    best_rf = best_model_results['model']
    
    print("\nMaking predictions on test set...")
    try:
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
        if 'Temparature' in test_df.columns:
            test_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
        
        # Prepare test features (match training preprocessing)
        test_features = test_df.drop(['id'], axis=1)
        test_features['NP_ratio'] = test_features['Nitrogen'] / (test_features['Phosphorous'] + 1e-8)
        test_features['NK_ratio'] = test_features['Nitrogen'] / (test_features['Potassium'] + 1e-8)
        test_features['Temp_Humidity'] = test_features['Temperature'] * test_features['Humidity']
        test_features = pd.get_dummies(test_features, columns=['Soil Type', 'Crop Type'])
        
        # Align columns with training data
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
        if 'Temparature' in train_df.columns:
            train_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
        train_features = train_df.drop(['id', 'Fertilizer Name'], axis=1)
        train_features['NP_ratio'] = train_features['Nitrogen'] / (train_features['Phosphorous'] + 1e-8)
        train_features['NK_ratio'] = train_features['Nitrogen'] / (train_features['Potassium'] + 1e-8)
        train_features['Temp_Humidity'] = train_features['Temperature'] * train_features['Humidity']
        train_features = pd.get_dummies(train_features, columns=['Soil Type', 'Crop Type'])
        
        missing_cols = set(train_features.columns) - set(test_features.columns)
        for col in missing_cols:
            test_features[col] = 0
        test_features = test_features[train_features.columns]
        
        # Normalize using training stats
        num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
                   'Potassium', 'Phosphorous', 'NP_ratio', 'NK_ratio', 'Temp_Humidity']
        train_num = train_features[num_cols].values
        X_mean = np.mean(train_num, axis=0)
        X_std = np.std(train_num, axis=0) + 1e-8
        test_num = test_features[num_cols].values
        test_features[num_cols] = (test_num - X_mean) / X_std
        
        # Get probabilities and top 3 predictions
        proba = best_rf.predict_proba(test_features.values)
        top3_preds = np.argsort(-proba, axis=1)[:, :3]
        test_preds_labels = [[label_map[pred] for pred in row] for row in top3_preds]
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Fertilizer Name': [" ".join(preds) for preds in test_preds_labels]
        })
        
        submission_path = 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"✅ Submission saved to {submission_path}")
        print("\nSample predictions:")
        print(submission.head())
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


# Import required libraries for evaluation
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt                                                                  
# After training your best model (best_rf), add this:

# Make predictions
y_pred = best_rf.predict(X_test)

# Now you can generate all evaluation metrics
print("\n=== Detailed Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_map, zero_division=0))

# Macro and Micro averaged F1 scores
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
print(f"\nMacro F1-score: {macro_f1:.4f}")
print(f"Micro F1-score: {micro_f1:.4f}")

# Macro Precision and Recall
macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
print(f"\nMacro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")

# Weighted Precision, Recall, and F1-score
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\nWeighted F1-score: {weighted_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")

# Plot normalized confusion matrix (recall per class)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=label_map,
    cmap='Blues',
    normalize='true',  # Normalize rows = recall per class
    values_format='.2f',  # Format values as decimals
    xticks_rotation=45
)
plt.title("Normalized Confusion Matrix (Recall per Class)")
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

class ROCCurveAnalyzer:
    def __init__(self, model, label_map):
        self.model = model
        self.label_map = label_map
        self.n_classes = len(label_map)
        
    def compute_class_probabilities(self, X):
        """Calculate class probabilities from Random Forest"""
        X = np.asarray(X)
        proba = np.zeros((X.shape[0], self.n_classes))
        
        for tree, feat_idxs in self.model.trees:
            X_subset = X[:, feat_idxs]
            preds = tree.predict(X_subset)
            proba[np.arange(X.shape[0]), preds] += 1
            
        return proba / len(self.model.trees)  # Normalize

    def plot_multiclass_roc(self, X_test, y_test, figsize=(12, 10)):
        """Plot ROC curves for all classes with improved visualization"""
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=range(self.n_classes))
        
        # Compute probabilities
        y_score = self.compute_class_probabilities(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        # Average and compute AUC
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=figsize)
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average (AUC = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                       'purple', 'brown', 'pink', 'gray', 'olive'])
        
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{self.label_map[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Analysis')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('multiclass_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc

def load_and_preprocess_data():
    """Enhanced data loading and preprocessing"""
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    
    # Handle column naming
    if 'Temparature' in train_df.columns:
        train_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    # Feature engineering
    X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
    y = train_df['Fertilizer Name'].astype('category').cat.codes
    label_map = train_df['Fertilizer Name'].astype('category').cat.categories.tolist()
    
    # Add interaction features
    X['NP_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1e-8)
    X['NK_ratio'] = X['Nitrogen'] / (X['Potassium'] + 1e-8)
    X['Temp_Humidity'] = X['Temperature'] * X['Humidity']
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
    
    # Normalization
    num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
               'Potassium', 'Phosphorous', 'NP_ratio', 'NK_ratio', 'Temp_Humidity']
    X[num_cols] = (X[num_cols] - X[num_cols].mean()) / (X[num_cols].std() + 1e-8)
    
    return X.values, y.values, label_map

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Improved train-test split with stratification"""
    from sklearn.model_selection import train_test_split as sk_split
    return sk_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    # Load and preprocess data
    X, y, label_map = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Initialize your trained model here
    # rf = RandomForest(...)
    # rf.fit(X_train, y_train)
    
    # Analyze ROC curves
    analyzer = ROCCurveAnalyzer(rf, label_map)
    roc_auc_scores = analyzer.plot_multiclass_roc(X_test, y_test)
    
    # Print AUC scores
    print("\nAUC Scores:")
    for class_idx, class_name in enumerate(label_map):
        print(f"{class_name}: {roc_auc_scores[class_idx]:.4f}")
    print(f"\nMacro-average AUC: {roc_auc_scores['macro']:.4f}")
    print(f"Micro-average AUC: {roc_auc_scores['micro']:.4f}")


# Import required libraries
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# 1. Save the best model
best_model = best_model_results['model']
joblib.dump(best_model, 'models/fertilizer_model.pkl')

# 2. Create and save label encoder (for fertilizer names)
# Using label_map from your original code which contains the original fertilizer names
le = LabelEncoder()
le.classes_ = label_map  # Directly assign the label_map (categories from your data)
joblib.dump(le, 'models/label_encoder.pkl')

print("\nModel and encoders saved successfully:")
print("- models/fertilizer_model.pkl")
print("- models/label_encoder.pkl")


!python --version

