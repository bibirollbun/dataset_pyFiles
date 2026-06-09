# Genetic Algorithm library
!pip install geneticalgorithm
# PSO library
!pip install pyswarm
!pip install geneticalgorithm lightgbm --quiet


import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif


train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
df = pd.merge(train_trans, train_identity, on='TransactionID', how='left')


# # Handle missing values
# df.fillna(-999, inplace=True)

# # One-hot encode categoricals (only existing columns)
# cat_cols = [col for col in ['ProductCD', 'DeviceType'] if col in df.columns]
# if cat_cols:
#     df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# # Downsample to 50K rows for speed
# df = df.iloc[:50000]
# X = df.drop(['isFraud', 'TransactionID'], axis=1)
# y = df['isFraud']


print("Columns in DataFrame:", df.columns.tolist())


# Part 1: Data Preprocessing (Fixed)
df.fillna(-999, inplace=True)

# Only encode columns that exist
cat_cols = [col for col in ['ProductCD', 'DeviceType'] if col in df.columns]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Reduces dimensionality

# Reduce memory usage (critical for large datasets)
for col in df.select_dtypes(include=['float64']):
    df[col] = df[col].astype('float32')
for col in df.select_dtypes(include=['int64']):
    df[col] = df[col].astype('int8')

# Downsample to 50K rows for faster experimentation
df = df.iloc[:50000]
X = df.drop(['isFraud', 'TransactionID'], axis=1)
y = df['isFraud']


# Use Mutual Information to reduce feature space
print("Selecting top 50% features by MI score...")
mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
top_features = np.where(mi_scores > np.quantile(mi_scores, 0.5))[0]  # Top 50%
X_reduced = X.iloc[:, top_features]
print(f"Reduced from {X.shape[1]} to {len(top_features)} features")


# Initialize fast LightGBM model
model = LGBMClassifier(class_weight='balanced', n_jobs=-1, random_state=42)

# Define fitness function (optimizes recall)
def fitness_function(feature_mask):
    selected = feature_mask > 0.5  # Threshold at 0.5
    if sum(selected) < 5:  # Require min 5 features
        return 1  # Penalize
    
    X_subset = X_reduced.iloc[:, selected]
    score = cross_val_score(model, X_subset, y, cv=3, scoring='recall', n_jobs=-1).mean()
    return -score  # Minimize negative recall

# GA hyperparameters (optimized for balance of speed/performance)
ga_params = {
    'max_num_iteration': 50,
    'population_size': 30,
    'mutation_probability': 0.25,
    'elit_ratio': 0.1,
    'parents_portion': 0.3,
    'crossover_probability': 0.7,
    'crossover_type': 'two_point',
    'max_iteration_without_improv': 15,
    'verbose': True
}


# Get selected features
best_mask = ga_model.output_dict['variable'] > 0.5
selected_features = X_reduced.columns[best_mask]

# Evaluate performance
final_recall = -ga_model.output_dict['function']

print(f"\nSelected {sum(best_mask)} features:")
print(selected_features.tolist())
print(f"Best Recall: {final_recall:.4f}")

# Optional: Compare to baseline
baseline = cross_val_score(model, X_reduced, y, cv=3, scoring='recall').mean()
print(f"Baseline (All Features): {baseline:.4f}")


model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

baseline_scores = cross_val_score(model, X, y, cv=3, scoring='recall')
print(f"Baseline Recall: {baseline_scores.mean():.4f}")


ga_recall = cross_val_score(model, X[ga_features], y, cv=3, scoring='recall').mean()
print(f"GA Model Recall: {ga_recall:.4f} (vs Baseline: {baseline_scores.mean():.4f})")


import matplotlib.pyplot as plt

metrics = ['Recall']
baseline = [baseline_scores.mean()]
ga_result = [ga_recall]
pso_result = [cross_val_score(model, X[pso_features], y, cv=3, scoring='recall').mean()]

plt.figure(figsize=(8, 4))
plt.bar(['Baseline', 'GA', 'PSO'], 
        [baseline[0], ga_result[0], pso_result[0]], 
        color=['gray', 'blue', 'orange'])
plt.title('Recall Comparison After Feature Selection')
plt.ylabel('Score')
plt.show()


from pyswarm import pso

def pso_fitness(feature_weights):
    selected = feature_weights > 0.5
    if sum(selected) == 0:
        return 1  # Penalize empty selections
    X_subset = X.iloc[:, selected]
    return -cross_val_score(model, X_subset, y, cv=2, scoring='recall').mean()

# Bounds for each feature (0=exclude, 1=include)
lb, ub = [0]*X.shape[1], [1]*X.shape[1]

# Run PSO (limited iterations for Kaggle)
best_weights, _ = pso(pso_fitness, lb, ub, swarmsize=20, maxiter=20)

# Get selected features
pso_features = X.columns[best_weights > 0.5]
print(f"PSO Selected Features ({len(pso_features)}):\n{pso_features.tolist()}")


import matplotlib.pyplot as plt

metrics = ['Recall']
baseline = [baseline_scores.mean()]
ga_result = [ga_recall]
pso_result = [cross_val_score(model, X[pso_features], y, cv=3, scoring='recall').mean()]

plt.figure(figsize=(8, 4))
plt.bar(['Baseline', 'GA', 'PSO'], 
        [baseline[0], ga_result[0], pso_result[0]], 
        color=['gray', 'blue', 'orange'])
plt.title('Recall Comparison After Feature Selection')
plt.ylabel('Score')
plt.show()

