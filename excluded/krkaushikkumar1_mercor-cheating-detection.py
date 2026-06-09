import pandas as pd
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import optuna
import warnings
import time
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

print("✓ All imports successful")


## 1. Data Loading and Validation

# Configuration
OUTPUT_FILE = "submission_tuned.csv"
RANDOM_STATE = 42

# Load data with validation
train_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
social_graph = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")
with open("/kaggle/input/mercor-cheating-detection/feature_metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"✓ Train Shape: {train_df.shape}")
print(f"✓ Test Shape: {test_df.shape}")
print(f"✓ Social Graph Edges: {social_graph.shape[0]}")
print(f"✓ Feature Metadata: {len(metadata)} features")

# Data validation
assert train_df['user_hash'].duplicated().sum() == 0, "Duplicate users in train set"
assert test_df['user_hash'].duplicated().sum() == 0, "Duplicate users in test set"
assert len(set(train_df['user_hash']) & set(test_df['user_hash'])) == 0, "Train/test overlap detected"
print("✓ Data validation passed")

# Show label distribution
print("\nLabel Distribution (Train):")
print(train_df['is_cheating'].value_counts(dropna=False))


## 2. Advanced Graph Feature Engineering

print("="*60)
print("Building Social Network Graph...")
print("="*60)

start_time = time.time()

# Build graph
G = nx.from_pandas_edgelist(social_graph, 'user_a', 'user_b')
print(f"Graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# 1. Degree Centrality
print("\nCalculating Degree Centrality...")
degree_dict = dict(G.degree())

# 2. PageRank
print("Calculating PageRank...")
pagerank_dict = nx.pagerank(G, alpha=0.85)

# 3. Clean Neighbor Ratio (Trust Propagation)
print("Calculating Clean Neighbor Ratio...")
clean_users = set(train_df[train_df['high_conf_clean'] == 1]['user_hash'])
print(f"  High-confidence clean users: {len(clean_users):,}")

clean_neighbor_ratio = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if not neighbors:
        clean_neighbor_ratio[node] = 0
    else:
        clean_count = sum(1 for nb in neighbors if nb in clean_users)
        clean_neighbor_ratio[node] = clean_count / len(neighbors)

# 4. Clustering Coefficient (Skipped for large graphs for performance)
print("Evaluating Clustering Coefficient...")
if G.number_of_nodes() < 50000:
    print("  Computing clustering coefficient...")
    clustering_dict = nx.clustering(G)
else:
    print(f"  Graph too large ({G.number_of_nodes():,} nodes), skipping clustering coeff")
    clustering_dict = {node: 0 for node in G.nodes()}

# Compile Graph Features DataFrame
graph_features = pd.DataFrame({
    'user_hash': list(degree_dict.keys()),
    'graph_degree': list(degree_dict.values()),
    'graph_pagerank': [pagerank_dict.get(k, 0) for k in degree_dict.keys()],
    'graph_clean_ratio': [clean_neighbor_ratio.get(k, 0) for k in degree_dict.keys()],
    'graph_clustering': [clustering_dict.get(k, 0) for k in degree_dict.keys()]
})

print(f"\nGraph Features Created: {graph_features.shape}")

# Merge with main datasets
train_df = train_df.merge(graph_features, on='user_hash', how='left')
test_df = test_df.merge(graph_features, on='user_hash', how='left')

# Fill NaNs (for isolated nodes not in graph)
for col in ['graph_degree', 'graph_pagerank', 'graph_clean_ratio', 'graph_clustering']:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

print(f"✓ Graph feature engineering completed in {time.time() - start_time:.1f}s")


## 3. Feature Selection and Data Preparation

# Extract features by type from metadata
numerical_cols = [col for col, meta in metadata.items() if meta['type'] == 'numeric']
binary_cols = [col for col, meta in metadata.items() if meta['type'] == 'binary']
graph_cols = ['graph_degree', 'graph_pagerank', 'graph_clean_ratio', 'graph_clustering']

feature_cols = numerical_cols + binary_cols + graph_cols

print(f"Feature Summary:")
print(f"  Numerical features: {len(numerical_cols)}")
print(f"  Binary features: {len(binary_cols)}")
print(f"  Graph features: {len(graph_cols)}")
print(f"  Total: {len(feature_cols)} features")

# Prepare training data
print("\n" + "="*60)
print("Preparing Training Data...")
print("="*60)

train_df_labeled = train_df.copy()

# Label imputation: high_conf_clean=1 with NaN is_cheating -> is_cheating=0
clean_mask = (train_df_labeled['high_conf_clean'] == 1) & (train_df_labeled['is_cheating'].isna())
print(f"Assigning labels from high_conf_clean: {clean_mask.sum():,} rows")
train_df_labeled.loc[clean_mask, 'is_cheating'] = 0

# Keep only labeled rows
train_df_labeled = train_df_labeled.dropna(subset=['is_cheating'])

X = train_df_labeled[feature_cols].fillna(0)
y = train_df_labeled['is_cheating']
X_test = test_df[feature_cols].fillna(0)

print(f"\nTraining Data:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Test shape: {X_test.shape}")

print(f"\nTarget Distribution:")
print(f"  Class 0 (Not Cheating): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  Class 1 (Cheating): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")
print(f"  Class Imbalance Ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")


## 4. Hyperparameter Tuning with Optuna

print("="*60)
print("Hyperparameter Tuning with Optuna")
print("="*60)

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist'  # Faster training
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        preds = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, preds)
        scores.append(score)
    
    return np.mean(scores)

print("\nStarting Optuna optimization with 20 trials...")
print("(This may take 20-30 minutes)\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n" + "="*60)
print("Optimization Results")
print("="*60)
print(f"Best Trial: #{study.best_trial.number}")
print(f"Best AUC: {study.best_trial.value:.4f}")
print(f"\nBest Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")


## 5. Final Model Training with Cross-Validation

print("="*60)
print("Training Final Models (5-Fold CV)")
print("="*60)

# Use best parameters from Optuna
best_params = study.best_trial.params.copy()
best_params['n_jobs'] = -1
best_params['eval_metric'] = 'auc'
best_params['tree_method'] = 'hist'
best_params['random_state'] = RANDOM_STATE

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))
fold_scores = []

print(f"\nTraining {FOLDS} models...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{FOLDS}")
    print("-" * 40)
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**best_params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Validation predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / FOLDS
    
    # Score
    fold_auc = roc_auc_score(y_val, val_preds)
    fold_scores.append(fold_auc)
    
    print(f"\n✓ Fold {fold+1} AUC: {fold_auc:.4f}\n")

# Overall performance
overall_auc = roc_auc_score(y, oof_preds)
mean_auc = np.mean(fold_scores)
std_auc = np.std(fold_scores)

print("="*60)
print("Cross-Validation Results")
print("="*60)
print(f"Fold AUCs: {[f'{score:.4f}' for score in fold_scores]}")
print(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
print(f"Overall CV AUC: {overall_auc:.4f}")
print("="*60)


## 6. Generate and Validate Submission

print("="*60)
print("Generating Submission File")
print("="*60)

# Create submission dataframe
submission = pd.DataFrame({
    'user_hash': test_df['user_hash'],
    'is_cheating': test_preds
})

# Validation checks
print("\nValidating submission format...")
assert submission.shape[1] == 2, "Submission should have 2 columns"
assert submission.shape[0] == len(test_df), f"Expected {len(test_df)} rows, got {submission.shape[0]}"
assert list(submission.columns) == ['user_hash', 'is_cheating'], "Incorrect column names"
assert submission['is_cheating'].min() >= 0 and submission['is_cheating'].max() <= 1, "Predictions out of range [0, 1]"
assert submission['user_hash'].duplicated().sum() == 0, "Duplicate user_hash in submission"
print("✓ Submission format validated")

# Print statistics
print("\nSubmission Statistics:")
print(f"  Total rows: {len(submission):,}")
print(f"  Min probability: {submission['is_cheating'].min():.6f}")
print(f"  Max probability: {submission['is_cheating'].max():.6f}")
print(f"  Mean probability: {submission['is_cheating'].mean():.6f}")
print(f"  Median probability: {submission['is_cheating'].median():.6f}")
print(f"  Std Dev: {submission['is_cheating'].std():.6f}")

print("\nSample Predictions:")
print(submission.head(10))

# Save submission
output_path = OUTPUT_FILE
submission.to_csv(output_path, index=False)
print(f"\n✓ Submission saved to: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

print("\n" + "="*60)
print("Pipeline Complete - Ready for Submission!")
print("="*60)

