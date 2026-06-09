#


# List workspace files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Imports
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Configs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Load Data
train_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
graph_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")
sample_submission = pd.read_csv("/kaggle/input/mercor-cheating-detection/sample_submission.csv")


# Print train columns
display("train_df columns:", train_df.columns)


# Summary
print('Summary about train_df')
train_df.info()

print('\nSummary about test_df')
test_df.info()


# Distribution of Target
train_df.is_cheating.value_counts(normalize=True)

# High imbalance dataset


# Features
feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
print(f"Detected {len(feature_cols)} features: {feature_cols}")


# Use high_conf_nf=1 rows as weak negatives (assumed not fraud) to enlarge the training set
train_df.loc[train_df['high_conf_clean'] == 1, 'is_fraud'] = 0

# Distribution of Target (Preview)
train_df.is_fraud.value_counts(normalize=True)


# Split Labeled / Unlabeled
labeled_df = train_df[train_df['is_fraud'].notnull()].copy()
unlabeled_df = train_df[train_df['is_fraud'].isnull()].copy()


# Feature Scaling
scaler = StandardScaler()
labeled_df[feature_cols] = scaler.fit_transform(labeled_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])
if not unlabeled_df.empty:
    unlabeled_df[feature_cols] = scaler.transform(unlabeled_df[feature_cols])


# Graph Features
graph_df.columns = ["source", "target"]
G = nx.from_pandas_edgelist(graph_df, 'source', 'target', create_using=nx.DiGraph())
degree_dict = dict(G.degree())
for df in [labeled_df, test_df, unlabeled_df]:
    if not df.empty:
        df['degree'] = df['user_hash'].map(degree_dict).fillna(0)


# Prepare Data
X = labeled_df[feature_cols + ['degree']]
y = labeled_df['is_fraud'].astype(int)
test_X = test_df[feature_cols + ['degree']]


# Train LightGBM with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
val_preds_full = np.zeros(len(X))
test_preds = np.zeros(len(test_df))
cv_models = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )
    cv_models.append(model)
    
    val_preds_full[val_idx] = model.predict_proba(X_val)[:,1]
    test_preds += model.predict_proba(test_X)[:,1] / skf.n_splits


# Optional: Semi-Supervised Pseudo-Labeling
if not unlabeled_df.empty:
    # Tune pseudo treshold
    pseudo_treshold_list = np.arange(0.5, 1.0, 0.05)
    fitted_models = dict()
    
    for pseudo_threshold in pseudo_treshold_list:
        print(f'LOGS FOR PS={pseudo_threshold}')
        
        pseudo_probs = np.zeros(len(unlabeled_df))
        for model in cv_models:
            pseudo_probs += model.predict_proba(unlabeled_df[feature_cols + ['degree']])[:,1]
        pseudo_probs /= skf.n_splits

        pseudo_labeled_df = unlabeled_df[pseudo_probs > pseudo_threshold].copy()
        pseudo_labeled_df['is_fraud'] = 1
        
        if not pseudo_labeled_df.empty:
            semi_X_full = pd.concat([X, pseudo_labeled_df[feature_cols + ['degree']]])
            semi_y_full = pd.concat([y, pseudo_labeled_df['is_fraud'].astype(int)])
            
            X_train, X_val, y_train, y_val = train_test_split(
                semi_X_full, semi_y_full, test_size=0.2, stratify=semi_y_full, random_state=SEED
            )
            
            semi_model = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED
            )
            
            semi_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)]
            )
            fitted_models[round(pseudo_threshold, 2)] = semi_model
            print()

# Use the best model
# if not unlabeled_df.empty:
#     optimized_model = fitted_models[0.75] # From logs
#     test_preds = optimized_model.predict_proba(test_X)[:,1]
#     val_preds_full = optimized_model.predict_proba(X)[:,1]  # for threshold optimization


# Cost-Based Threshold Optimization
def cost_function(thresholds, y_true, y_prob):
    auto_pass, manual_review = thresholds
    y_pred = np.zeros_like(y_true)
    y_pred[y_prob >= manual_review] = 2  # auto-block
    y_pred[(y_prob >= auto_pass) & (y_prob < manual_review)] = 1  # manual review
    
    cost = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1:  # Fraud
            if pred == 0: cost += 600   # FN
            elif pred == 1: cost += 5   # Manual review TP
            else: cost += 0              # Auto-block TP
                
        else:  # Not fraud
            if pred == 0: cost += 0
            elif pred == 1: cost += 150 # Manual review FP
            else: cost += 300            # Auto-block FP (Edit: "False Positive in auto-block region: $500 -> $300")

    return cost

# Use validation predictions for threshold optimization
res = minimize(cost_function, [0.3, 0.7], args=(y, val_preds_full), bounds=[(0,1),(0,1)])
auto_pass_thr, manual_review_thr = res.x
print(f"Optimal thresholds: auto_pass={auto_pass_thr:.3f}, manual_review={manual_review_thr:.3f}")


# Generate Submission
submission = pd.DataFrame({
    'user_hash': test_df['user_hash'],
    'prediction': test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved: submission.csv")
!head submission.csv


#

