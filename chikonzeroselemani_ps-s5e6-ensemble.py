import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
external = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# Combine and preprocess data
overall_train = pd.concat([train, external], ignore_index=True)
overall_train = overall_train.drop_duplicates()
x = overall_train.drop(columns=["Fertilizer Name"])
y = overall_train["Fertilizer Name"]

# Encode features and target
encoder = LabelEncoder()
target_encoder = LabelEncoder()
categorical = x.select_dtypes(include=['object']).columns

y_encoded = target_encoder.fit_transform(y)
for cat in categorical:
    x[cat] = encoder.fit_transform(x[cat])
    test[cat] = encoder.transform(test[cat])

# Create validation set for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Display dataset information
print(f"Training data shape: {x.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {test.shape}")
print(f"Number of classes: {len(np.unique(y_encoded))}")


def map_at_3(y_true, y_pred_proba, k=3):
    """Calculate Mean Average Precision at K (MAP@K) metric"""
    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    for i in range(len(y_true)):
        # Get indices of top k predictions
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
        if y_true[i] in top_k_preds:
            # Find position of true label in top k predictions
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1
            map_score += 1.0 / rank
    return map_score / len(y_true)

def log_map_at_3(y_true, y_pred_proba, k=3):
    """Log MAP@3 metric with additional information"""
    score = map_at_3(y_true, y_pred_proba, k)
    print(f"MAP@{k}: {score:.6f}")
    return score


xgb_params = {
    'tree_method': 'hist',
    'n_estimators': 5000,
    'objective': 'multi:softprob',
    'random_state': 32,
    'enable_categorical': True,
    'verbosity': 0,
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    "device": "cuda",
    'n_jobs': -1,
    'learning_rate': 0.1,
    'num_class': 7,
    'lambda': 0.05656209749983576,
    'alpha': 5.620898657099113,
    'colsample_bytree': 0.2587327850345624, 
    'subsample': 0.8276149323901826,
    'max_depth': 20,
    'min_child_weight': 10
    }

lgb_params = {
    'objective': 'multiclass',
    'num_class': 7,
    'device': 'gpu',
    'colsample_bytree': 0.4366677273946288,
    'learning_rate': 0.026164161953515117,
    'max_depth': 12,
    'min_child_samples': 67,
    'n_estimators': 10000,
    'n_jobs': -1,
    'num_leaves': 243,
    'random_state': 42,
    'reg_alpha': 6.38288560443373,
    'reg_lambda': 9.392999314379155,
    'subsample': 0.7989164499431718,
    'verbose': -1
}
cat_params = {
    'verbose': 0,
    'random_state': 42,
    'cat_features': x.columns.tolist(),
    'early_stopping_rounds': 50,
    'eval_metric': "MultiClass",
    'n_estimators': 5000,
    'objective': 'MultiClass', 
    'learning_rate': 0.1,
    "task_type": "GPU",
    'l2_leaf_reg': 0.12185512372394472,
    'bagging_temperature': 0.2119744763488629,
    'random_strength': 1.8864063201163634,
    'depth': 5,
    'min_data_in_leaf': 2,

}

print("Model parameters configured ")


n_classes = 7
n_splits=5
state=42
early_stop=50

# Initialize prediction matrices
oof_xgb = np.zeros((len(x), n_classes))
oof_lgb = np.zeros((len(x), n_classes))
oof_cat = np.zeros((len(x), n_classes))
fold_map_xgb = []
fold_map_lgb = []
fold_map_cat = []
test_preds_xgb = np.zeros((len(test), n_classes))
test_preds_lgb = np.zeros((len(test), n_classes))
test_preds_cat = np.zeros((len(test), n_classes))

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=state
)

print(f"Starting {n_splits}-fold StratifiedKFold")
print("=" * 70)

for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y_encoded)):
    print(f"\nFold {fold+1}/{n_splits}")
    print("-" * 50)
    X_tr, X_vld = x.iloc[train_idx], x.iloc[valid_idx]
    y_tr, y_vld = y_encoded[train_idx], y_encoded[valid_idx]
    
    # XGBoost training
    print("Training XGBoost...")
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=0)
    xgb_proba = xgb.predict_proba(X_vld)
    oof_xgb[valid_idx] = xgb_proba
    test_preds_xgb += xgb.predict_proba(test) / n_splits
    xgb_map = map_at_3(y_vld, xgb_proba)
    fold_map_xgb.append(xgb_map)
    print(f"XGBoost MAP@3: {xgb_map:.6f}")
    
    # LightGBM training
    print("\nTraining LightGBM...")
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_vld, y_vld)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(early_stop, verbose=False)],
    )
    lgb_proba = lgb.predict_proba(X_vld)
    oof_lgb[valid_idx] = lgb_proba
    test_preds_lgb += lgb.predict_proba(test) / n_splits
    lgb_map = map_at_3(y_vld, lgb_proba)
    fold_map_lgb.append(lgb_map)
    print(f"LightGBM MAP@3: {lgb_map:.6f}")
    
    # CatBoost training
    print("\nTraining CatBoost...")
    cat = CatBoostClassifier(**cat_params)
    cat.fit(
        X_tr, y_tr,
        eval_set=(X_vld, y_vld),
        verbose=0
    )
    cat_proba = cat.predict_proba(X_vld)
    oof_cat[valid_idx] = cat_proba
    test_preds_cat += cat.predict_proba(test) / n_splits
    cat_map = map_at_3(y_vld, cat_proba)
    fold_map_cat.append(cat_map)
    print(f"CatBoost MAP@3: {cat_map:.6f}")
    
    # Fold summary
    print("\n" + "=" * 50)

# Calculate overall metrics
overall_xgb_map = map_at_3(y_encoded, oof_xgb)
overall_lgb_map = map_at_3(y_encoded, oof_lgb)
overall_cat_map = map_at_3(y_encoded, oof_cat)
overall_avg_map = (overall_xgb_map + overall_lgb_map + overall_cat_map) / 3

print("\n" + "=" * 70)
print("Cross-Validation Complete!")
print("=" * 70)
print(f"\nOverall XGBoost MAP@3: {overall_xgb_map:.6f}")
print(f"Overall LightGBM MAP@3: {overall_lgb_map:.6f}")
print(f"Overall CatBoost MAP@3: {overall_cat_map:.6f}")
print("=" * 70)


# Create meta-features
meta_train = np.hstack([oof_xgb, oof_lgb, oof_cat])
print(f"\nMeta features shape: {meta_train.shape}")
print(f"Target labels shape: {y_encoded.shape}")

# Train meta-model
print("\nTraining Logistic Regression meta-model...")
meta_model = LogisticRegression(
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=2000,
    C=0.1,
    random_state=state,
    n_jobs=-1
)
meta_model.fit(meta_train, y_encoded)

# Evaluate meta-model
meta_train_proba = meta_model.predict_proba(meta_train)
meta_map = map_at_3(y_encoded, meta_train_proba)
print(f"Meta-model MAP@3 on training data: {meta_map:.6f}")

# Prepare test predictions
meta_test = np.hstack([test_preds_xgb, test_preds_lgb, test_preds_cat])
final_test_proba = meta_model.predict_proba(meta_test)

# Generate submission
top3_indices = np.argsort(-final_test_proba, axis=1)[:, :3]
top3_labels = target_encoder.inverse_transform(top3_indices.ravel())
top3_labels = top3_labels.reshape(len(test), 3)

submission = pd.DataFrame({
    "id": test.index,
    "Fertilizer Name": [" ".join(row) for row in top3_labels]
})
submission.to_csv("submission.csv", index=False)
print("\nsubmission created!")
print(submission.head())

