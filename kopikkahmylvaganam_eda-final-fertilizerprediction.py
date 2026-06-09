import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, log_loss, matthews_corrcoef
from sklearn.preprocessing import label_binarize


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("\nTrain Info:")
print(train_df.info())


# Missing values
print("\nMissing Values in Train:")
print(train_df.isnull().sum())


# Target variable distribution
plt.figure(figsize=(10,5))
sns.countplot(x='Fertilizer Name', data=train_df, order=train_df['Fertilizer Name'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Fertilizer Name Distribution")
plt.show()

# Soil Type distribution
plt.figure(figsize=(8,4))
sns.countplot(x='Soil Type', data=train_df, order=train_df['Soil Type'].value_counts().index)
plt.title("Soil Type Distribution")
plt.show()

# Crop Type distribution
plt.figure(figsize=(8,4))
sns.countplot(x='Crop Type', data=train_df, order=train_df['Crop Type'].value_counts().index)
plt.title("Crop Type Distribution")
plt.show()

# Numeric columns distribution
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f"{col} Distribution")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(train_df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Load
#train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

# 1. Basic Info
print("Dataset Shape:", train_df.shape)
print("\nColumn Names:", train_df.columns.tolist())
print("\nDataset Info:")
print(train_df.info())
print("\nFirst 5 Rows:")
print(train_df.head())

# 2. Check Missing Values
print("\nMissing Values per Column:")
print(train_df.isnull().sum())

# 3. Target Variable (Fertilizer Name) Overview
fertilizer_counts = train_df['Fertilizer Name'].value_counts()
print("\nUnique Fertilizers:", train_df['Fertilizer Name'].nunique())
print("\nFertilizer Counts:")
print(fertilizer_counts)

# Pie chart for Fertilizer distribution
plt.figure(figsize=(8, 8))
fertilizer_counts.plot.pie(autopct="%1.1f%%", startangle=140, cmap='tab20')
plt.title("Fertilizer Distribution (Pie Chart)")
plt.ylabel("")
plt.show()

# Countplot for Fertilizer distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='Fertilizer Name', order=fertilizer_counts.index, palette='Set2')
plt.xticks(rotation=45)
plt.title("Fertilizer Distribution (Countplot)")
plt.show()

# 4. Categorical Features Overview
print("\nUnique Soil Types:", train_df['Soil Type'].nunique())
print(train_df['Soil Type'].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=train_df, x='Soil Type', order=train_df['Soil Type'].value_counts().index, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Soil Type Distribution")
plt.show()

print("\nUnique Crop Types:", train_df['Crop Type'].nunique())
print(train_df['Crop Type'].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=train_df, x='Crop Type', order=train_df['Crop Type'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.title("Crop Type Distribution")
plt.show()

# 5. Numerical Feature Distributions
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col], kde=True, color='blue')
    plt.title(f"{col} Distribution")
    plt.show()

# 6. Scatter Plots (pairwise relationships with target color)
sns.pairplot(train_df[num_cols + ['Fertilizer Name']], hue='Fertilizer Name', palette='tab10')
plt.suptitle("Pairwise Scatter Plots by Fertilizer", y=1.02)
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = train_df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# 8. Boxplots to detect outliers
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Fertilizer Name', y=col, data=train_df, palette='Set3')
    plt.xticks(rotation=45)
    plt.title(f"{col} by Fertilizer Type (Boxplot)")
    plt.show()

print("\nEDA Completed Successfully âœ…")



# Mapping categorical values
soil_types = train_df['Soil Type'].unique()
crop_types = train_df['Crop Type'].unique()
fertilizers = train_df['Fertilizer Name'].unique()

soil_type_map = {soil: idx for idx, soil in enumerate(soil_types)}
crop_type_map = {crop: idx for idx, crop in enumerate(crop_types)}
fertilizer_map = {fert: idx for idx, fert in enumerate(fertilizers)}
reverse_fertilizer_map = {idx: fert for fert, idx in fertilizer_map.items()}

train_df['Soil Type'] = train_df['Soil Type'].map(soil_type_map)
train_df['Crop Type'] = train_df['Crop Type'].map(crop_type_map)
train_df['Fertilizer Name'] = train_df['Fertilizer Name'].map(fertilizer_map)

test_df['Soil Type'] = test_df['Soil Type'].map(soil_type_map).fillna(0).astype(int)
test_df['Crop Type'] = test_df['Crop Type'].map(crop_type_map).fillna(0).astype(int)

# Extract features & target
X = train_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']].values
y = train_df['Fertilizer Name'].values
X_test = test_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']].values

# Normalization
def normalize(X, means=None, stds=None):
    X_norm = X.copy().astype(float)
    indices = [0, 1, 2, 5, 6, 7]  # numeric feature indices
    if means is None or stds is None:
        means = np.mean(X[:, indices], axis=0)
        stds = np.std(X[:, indices], axis=0) + 1e-8
    for i, idx in enumerate(indices):
        X_norm[:, idx] = (X_norm[:, idx] - means[i]) / stds[i]
    return X_norm, means, stds

X, means, stds = normalize(X)
X_test, _, _ = normalize(X_test, means, stds)

# Train-validation split
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx, val_idx = indices[:train_size], indices[train_size:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


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
        weighted_counts = np.zeros(self.n_classes)
        for i in range(len(y)):
            weighted_counts[y[i]] += weights[i]
        probs = weighted_counts / (np.sum(weighted_counts) + 1e-8)
        return -np.sum(probs * np.log2(probs + 1e-8))

    def information_gain(self, X, y, feature_idx, threshold, weights=None):
        parent_entropy = self.entropy(y, weights)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        if n_left < 1 or n_right < 1:
            return 0
        left_entropy = self.entropy(y[left_mask], weights[left_mask] if weights is not None else None)
        right_entropy = self.entropy(y[right_mask], weights[right_mask] if weights is not None else None)
        n = len(y)
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def find_best_split(self, X, y, weights=None):
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
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
            class_counts = np.bincount(y, minlength=self.n_classes)
            return DecisionTreeNode(value=np.argmax(class_counts), class_counts=class_counts)
        feature_idx, threshold = self.find_best_split(X, y, weights)
        if feature_idx is None:
            class_counts = np.bincount(y, minlength=self.n_classes)
            return DecisionTreeNode(value=np.argmax(class_counts), class_counts=class_counts)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1, weights[left_mask] if weights is not None else None)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1, weights[right_mask] if weights is not None else None)
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
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_sample[:, feature_indices]
            tree.fit(X_subset, y_sample)
            self.trees.append((tree, feature_indices))

    def predict_proba(self, X):
        n_classes = len(np.unique(y))
        proba_sum = np.zeros((X.shape[0], n_classes))
        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            proba = tree.predict_proba(X_subset)
            proba_sum += proba
        return proba_sum / self.n_trees

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


models = [
    ("DecisionTree depth=5", DecisionTree(max_depth=5)),
    ("DecisionTree depth=7", DecisionTree(max_depth=7)),
    ("RandomForest trees=10 depth=5", RandomForest(n_trees=10, max_depth=5)),
    ("RandomForest trees=20 depth=7", RandomForest(n_trees=20, max_depth=7)),
    ("RandomForest trees=30 depth=9", RandomForest(n_trees=30, max_depth=9))
]

for name, model in models:
    model.fit(X_train, y_train)
    print(f"âœ… {name} trained.")



results = []
n_classes = len(np.unique(y_train))
y_val_bin = label_binarize(y_val, classes=list(range(n_classes)))

for name, model in models:
    y_val_proba = model.predict_proba(X_val)
    y_val_pred = np.argmax(y_val_proba, axis=1)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    rec = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_proba)
    mcc = matthews_corrcoef(y_val, y_val_pred)
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_val_proba.ravel())
    roc_auc = auc(fpr, tpr)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'LogLoss': ll,
        'MCC': mcc,
        'AUC': roc_auc,
        'FPR': fpr,
        'TPR': tpr,
        'ConfusionMatrix': cm
    })

metrics_df = pd.DataFrame(results)
print(metrics_df[['Model','Accuracy','Precision','Recall','F1 Score','LogLoss','MCC','AUC']])



for r in results:
    plt.figure(figsize=(6,4))
    sns.heatmap(r['ConfusionMatrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {r['Model']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



plt.figure(figsize=(8,6))
for r in results:
    plt.plot(r['FPR'], r['TPR'], label=f"{r['Model']} (AUC = {r['AUC']:.3f})")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.show()

print("\nğŸ�† Best Model by AUC Score:")
print(metrics_df.loc[metrics_df['AUC'].idxmax()])


# Select best model by AUC
best_model_index = metrics_df['AUC'].idxmax()
best_model_name = metrics_df.loc[best_model_index, 'Model']
best_model = models[best_model_index][1]  # Model object

print(f"\nâœ… Using Best Model for Test Prediction: {best_model_name}")

# Predict probabilities for test dataset
y_test_proba = best_model.predict_proba(X_test)

# Get Top-3 predicted fertilizers for each test sample
top3_preds = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
top3_fertilizers = [[reverse_fertilizer_map[idx] for idx in pred] for pred in top3_preds]

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top3_fertilizers]
})

# Save to CSV
submission_file = "/kaggle/working/submission_best_model.csv"
submission.to_csv(submission_file, index=False)

print(f"ğŸ“„ Submission file saved successfully: {submission_file}")
print(submission.head())


import joblib
model_file = "/kaggle/working/best_model.joblib"
joblib.dump(best_model, model_file)

print(f"ğŸ’¾ Best model saved successfully at: {model_file}")





