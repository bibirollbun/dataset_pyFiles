import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from math import sqrt, log
from multiprocessing import Pool
from time import time
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt






df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
ids = test_df['id']


# Fix typo if exists
if 'Temparature' in df.columns:
    df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

df.drop(columns=['id'], inplace=True)


df.info()


df.columns


df.shape


df.head()


df.tail()


df.describe()


# Data types and missing values
print("\nData Types and Missing Values:")
train_info = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Unique Values': df.nunique()
})
display(train_info)



target_col = 'Fertilizer Name'
# Countplot with percentages
plt.figure(figsize=(14, 6))
ax = sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index)
plt.title('Distribution of Fertilizer Types (Target Variable)', fontsize=16)
plt.xlabel('Fertilizer Name', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)


#Pie Chart for Fertilizer Distribution
plt.figure(figsize=(8, 8))
df['Fertilizer Name'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Fertilizer Distribution (Pie Chart)')
plt.ylabel('')
plt.show()



# Visualization of categorical features
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Soil Type')
plt.title('Distribution of Soil Types')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Crop Type')
plt.title('Distribution of Crop Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Pairplot for numerical features

# Define numerical columns (excluding 'id' and categorical columns)
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


sns.pairplot(df[numerical_cols])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# Boxplots for numerical features vs target

# Define numerical columns (excluding 'id' and categorical columns)
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


plt.figure(figsize=(15, 20))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=df, x='Fertilizer Name', y=col)
    plt.title(f'{col} by Fertilizer Type')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Correlation heatmap
# Define numerical columns (excluding 'id' and categorical columns)
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


plt.figure(figsize=(10, 8))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# 2. Enhanced Histograms with KDE (without sketch)
# Define numerical columns (excluding 'id' and categorical columns)
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


print("\n=== KDE Plot Insights ===")
print("Kernel Density Estimates show the probability density of numerical features.")
print("Peaks indicate common values, while spread shows variability.\n")

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, stat='density', linewidth=0)
    plt.title(f'KDE Plot of {col}')
    plt.show()


# 3. Scatter Plots with Hue

# Define numerical columns (excluding 'id' and categorical columns)
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


print("\n=== Scatter Plot Relationships ===")
print("Scatter plots reveal relationships between numerical features colored by fertilizer type.")
sns.pairplot(df, vars=numerical_cols[:4], hue='Fertilizer Name', height=3)
plt.suptitle('Scatter Matrix with Fertilizer Hue', y=1.02)
plt.show()


# 4. Line Plots (Averaged by Fertilizer)
plt.figure(figsize=(12, 6))
for col in numerical_cols:
    sns.lineplot(data=df, x='Fertilizer Name', y=col, ci=None)
plt.title('Average Values by Fertilizer Type (Line Plot)')
plt.xticks(rotation=45)
plt.legend(numerical_cols, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# 5. Violin Plots
print("\n=== Violin Plot Insights ===")
print("Violin plots show the distribution, median, and interquartile range of each feature by fertilizer type.")

plt.figure(figsize=(15, 20))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.violinplot(data=df, x='Fertilizer Name', y=col)
    plt.title(f'Violin Plot of {col} by Fertilizer')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 6. Swarm Plots (sample for better performance)
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df.sample(200), x='Fertilizer Name', y='Temperature', hue='Soil Type')
plt.title('Swarm Plot of Temperature by Fertilizer (Sample)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# 7. Strip Plots
plt.figure(figsize=(12, 6))
sns.stripplot(data=df, x='Fertilizer Name', y='Nitrogen', jitter=True, alpha=0.5)
plt.title('Strip Plot of Nitrogen by Fertilizer')
plt.xticks(rotation=45)
plt.show()


# 8. Bubble Plot (using scatter plot)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df.sample(200), x='Nitrogen', y='Phosphorous', 
                size='Potassium', hue='Fertilizer Name', alpha=0.7)
plt.title('Bubble Plot: N vs P with K as Size (Sample)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 9. Radar Chart
print("\n=== Radar Chart ===")
print("Radar chart comparing average feature values across fertilizer types.")

categories = numerical_cols
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)

for fert in df['Fertilizer Name'].unique()[:3]:  # Plot first 3 for clarity
    values = df[df['Fertilizer Name']==fert][categories].mean().values.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, label=fert)
    ax.fill(angles, values, alpha=0.1)

plt.title('Radar Chart: Feature Means by Fertilizer', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()


# 10. ECDF Plots
print("\n=== ECDF Insights ===")
print("Empirical CDF shows the cumulative distribution of values for each feature.")

for col in numerical_cols[:3]:  # First 3 for example
    plt.figure(figsize=(8, 5))
    sns.ecdfplot(data=df, x=col, hue='Fertilizer Name')
    plt.title(f'ECDF of {col} by Fertilizer')
    plt.show()


# 11. 3D Scatter Plot
print("\n=== 3D Relationships ===")
print("3D visualization of Nitrogen, Phosphorous and Potassium relationships.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sample = df.sample(200)  # Sample for performance

xs = sample['Nitrogen']
ys = sample['Phosphorous']
zs = sample['Potassium']
scatter = ax.scatter(xs, ys, zs, c=sample['Temperature'], cmap='viridis')

ax.set_xlabel('Nitrogen')
ax.set_ylabel('Phosphorous')
ax.set_zlabel('Potassium')
plt.colorbar(scatter, label='Temperature')
plt.title('3D Scatter: N-P-K Colored by Temperature')
plt.show()


# 12. Density Plot Matrix
g = sns.PairGrid(df[numerical_cols[:4]])
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
plt.suptitle('Density Plot Matrix', y=1.02)
plt.show()


print("\n=== Advanced Correlation ===")
print("Heatmaps for feature correlations per Fertilizer class.")

# First, get the unique fertilizer names from your DataFrame
fertilizers = df['Fertilizer Name'].unique()

n_cols = 3
n_rows = (len(fertilizers) // n_cols) + 1

plt.figure(figsize=(6*n_cols, 5*n_rows))
for i, fert in enumerate(fertilizers, 1):
    subset = df[df['Fertilizer Name'] == fert][numerical_cols].dropna()
    if len(subset) < 5:
        continue  # Skip very small subsets
        
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(
        subset.corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        mask=np.triu(np.ones_like(subset.corr()))  # Hide upper triangle
    )
    plt.title(f'{fert} (n={len(subset)})')
    
plt.tight_layout()
plt.show()




# --- 1. Handle Categorical Features ---
# One-hot encode (safer than label encoding for non-ordinal categories)
soil_dummies = pd.get_dummies(df['Soil Type'], prefix='soil', dtype=int)
crop_dummies = pd.get_dummies(df['Crop Type'], prefix='crop', dtype=int)

# --- 2. Handle Numerical Features ---
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_numerical = df[numerical_cols].values  # Convert to numpy array early for efficiency

# --- 3. Combine Features ---
X = np.concatenate([X_numerical, soil_dummies, crop_dummies], axis=1)

# --- 4. Label Encoding ---
# Use factorize() instead of cat.codes to ensure consistent encoding
y, y_labels = pd.factorize(df['Fertilizer Name'])
print("Label mapping:", dict(enumerate(y_labels)))  # Debug: Show class mapping

# --- 5. Train-Test Split ---
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.6 * len(X))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- 6. Normalization ---
# Only normalize numerical columns (cols 0-5) to avoid distorting one-hot encoded values
X_train_num = X_train[:, :6]
X_test_num = X_test[:, :6]

X_mean = X_train_num.mean(axis=0)
X_std = X_train_num.std(axis=0) + 1e-8  # Avoid division by zero

X_train[:, :6] = (X_train_num - X_mean) / X_std
X_test[:, :6] = (X_test_num - X_mean) / X_std

print("Training shape:", X_train.shape, "Test shape:", X_test.shape)


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
    df.drop(columns=['id'], inplace=True)
    
    # Better categorical encoding (one-hot for NN, label for others)
    df = pd.get_dummies(df, columns=['Soil Type', 'Crop Type'])
    
    # Label encode target
    df['Fertilizer Name'] = df['Fertilizer Name'].astype('category').cat.codes
    
    # Separate features and target
    X = df.drop(['Fertilizer Name'], axis=1).astype(float).values
    y = df['Fertilizer Name'].astype('category').cat.codes.values
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Normalize features (using train stats)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
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
        
        for epoch in range(1, self.epochs+1):
            # Forward pass
            z1 = X.dot(self.W1) + self.b1
            a1 = self.relu(z1)
            z2 = a1.dot(self.W2) + self.b2
            a2 = self.softmax(z2)
            
            # Backpropagation
            dz2 = a2 - y_onehot
            dW2 = a1.T.dot(dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = dz2.dot(self.W2.T)
            dz1 = da1 * self.relu_derivative(a1)
            dW1 = X.T.dot(dz1)
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
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = a1.dot(self.W2) + self.b2
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
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

# --------------------------
# 3. Random Forest (Ensemble Method) - Optimized
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
        tree = DecisionTree(max_depth=self.max_depth, 
                          min_samples_split=self.min_samples_split)
        tree.fit(X[:, feat_idxs], y)
        return (tree, feat_idxs)
    
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features
        
        # Prepare all tree arguments
        tree_args = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
            tree_args.append((X_sample, y_sample, feat_idxs))
        
        # Train trees in parallel
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

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'model': model
    }

# --------------------------
# Main Execution - Optimized
# --------------------------

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    X, y = preprocess_data(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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



import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import Counter
from sklearn.metrics import accuracy_score
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --------------------------
# Decision Tree Implementation
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
    
    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])[:10]  # Check first 10 unique values
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if (len(y[left_idx]) < self.min_samples_split or 
                    len(y[right_idx]) < self.min_samples_split or
                    len(y[left_idx]) < self.min_samples_leaf or
                    len(y[right_idx]) < self.min_samples_leaf):
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
# Random Forest Implementation
# --------------------------
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=20, 
                max_features=None, min_samples_leaf=1, n_jobs=4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
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
        
        # Handle max_features parameter
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        elif self.max_features is None:
            self.max_features = n_features
        
        # Process trees in batches to reduce memory usage
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
    
    def predict(self, X):
        # Predict in batches to reduce memory usage
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
# Data Loading and Evaluation
# --------------------------
def load_and_preprocess():
    print("Loading and preprocessing data...")
    df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    
    if 'Temparature' in df.columns:
        df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    X = df.drop(['id', 'Fertilizer Name'], axis=1)
    y = df['Fertilizer Name'].astype('category').cat.codes
    
    X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
    num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, scaler

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load and preprocess data
    X, y, scaler = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    
    # Model configurations to test
    models = [
        {
            'name': 'RF1 - Balanced', 
            'params': {
                'n_trees': 100,
                'max_depth': 20,
                'min_samples_split': 10,
                'max_features': 'sqrt',
                'min_samples_leaf': 2,
                'n_jobs': 2
            }
        },
        {
            'name': 'RF2 - Regularized', 
            'params': {
                'n_trees': 80,
                'max_depth': 15,
                'min_samples_split': 15,
                'max_features': 0.6,
                'min_samples_leaf': 3,
                'n_jobs': 2
            }
        },
        {
            'name': 'RF3 - Efficient', 
            'params': {
                'n_trees': 120,
                'max_depth': 25,
                'min_samples_split': 5,
                'max_features': 'log2',
                'min_samples_leaf': 1,
                'n_jobs': 2
            }
        }
    ]
    
    results = {}
    for model in models:
        print("\n" + "="*60)
        print(f"EVALUATING {model['name']}")
        print(f"Parameters: {model['params']}")
        print("="*60)
        
        rf = RandomForest(
            n_trees=model['params']['n_trees'],
            max_depth=model['params']['max_depth'],
            min_samples_split=model['params']['min_samples_split'],
            max_features=model['params']['max_features'],
            min_samples_leaf=model['params']['min_samples_leaf'],
            n_jobs=model['params']['n_jobs']
        )
        
        start_time = time()
        rf.fit(X_train, y_train)
        train_time = time() - start_time
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[model['name']] = {
            'test_accuracy': acc,
            'training_time': train_time,
            'params': model['params'],
            'model': rf
        }
        
        print(f"\n{model['name']} Test Accuracy: {acc:.4f}")
        print("="*60)
    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"Training Time: {res['training_time']:.2f}s")
        print(f"Parameters: {res['params']}")
    
    best_model_name, best_model_results = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_rf = best_model_results['model']
    print("\n" + "="*50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_model_results['test_accuracy']:.4f}")
    print(f"Parameters: {best_model_results['params']}")


# Make predictions on test set
print("\nMaking predictions on test set...")
try:
    # Load test data
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    if 'Temparature' in test_df.columns:
        test_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    # Prepare test features
    test_features = test_df.drop(['id'], axis=1)
    test_features = pd.get_dummies(test_features, columns=['Soil Type', 'Crop Type'])
    
    # Load training data and create label map
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    if 'Temparature' in train_df.columns:
        train_df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    # Create label map first (FIXED: Correct column name 'Fertilizer Name')
    label_map = train_df['Fertilizer Name'].astype('category').cat.categories
    
    # Prepare training features for column alignment
    train_features = train_df.drop(['id', 'Fertilizer Name'], axis=1)
    train_features = pd.get_dummies(train_features, columns=['Soil Type', 'Crop Type'])
    
    # Align columns with training data
    missing_cols = set(train_features.columns) - set(test_features.columns)
    for col in missing_cols:
        test_features[col] = 0
    test_features = test_features[train_features.columns]
    
    # Scale numerical features
    num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    scaler = StandardScaler()
    scaler.fit(train_features[num_cols])  # Fit on training data
    test_features[num_cols] = scaler.transform(test_features[num_cols])
    
    # Get class probabilities from the Random Forest
    class_probabilities = np.zeros((len(test_features), len(label_map)))
    
    for tree, feat_idxs in best_rf.trees:
        # Get predictions for this tree (returns class indices)
        tree_preds = tree.predict(test_features.values[:, feat_idxs])
        
        # Convert predictions to one-hot encoded probabilities
        for i, pred in enumerate(tree_preds):
            class_probabilities[i, pred] += 1
    
    # Normalize probabilities (convert counts to probabilities)
    class_probabilities /= len(best_rf.trees)
    
    # Get top 3 predictions for each sample
    top3_preds = np.argsort(-class_probabilities, axis=1)[:, :3]
    
    # Convert back to original labels
    test_preds_labels = [[label_map[pred] for pred in row] for row in top3_preds]
    
    # Create submission with top 3 recommendations
    submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [" ".join(preds) for preds in test_preds_labels]  # Join with single spaces
})
    
    # Save submission
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"✅ Submission with top 3 recommendations saved to {submission_path}")
    print(submission.head())
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    raise


from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Classification Report (includes precision, recall, f1-score per class)
print("\n=== Detailed Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_map))

# 2. Macro/Micro F1 Scores
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print(f"\nMacro F1-score: {macro_f1:.4f}")
print(f"Micro F1-score: {micro_f1:.4f}")

# 3. Precision and Recall (macro-averaged)
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
print(f"\nMacro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")

# 4. Weighted Metrics (accounts for class imbalance)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
weighted_precision = precision_score(y_test, y_pred, average='weighted')
weighted_recall = recall_score(y_test, y_pred, average='weighted')
print(f"\nWeighted F1-score: {weighted_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")

# 5. Confusion Matrix (visual)
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred,
    display_labels=label_map,
    xticks_rotation=45,
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Gradient Boosting often performs better
from xgboost import XGBClassifier
xgb = XGBClassifier(scale_pos_weight=1)  # Auto-handles imbalance
xgb.fit(X_train, y_train)

