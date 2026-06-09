# =========================================================
# Enhanced Bank Marketing Binary Classification with LightGBM
# =========================================================

# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# 2. Load and Prepare Data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# 3. Enhanced EDA
print("Target Distribution:")
print(train['y'].value_counts(normalize=True))

plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=train)
plt.title("Target Distribution (0: No Subscription, 1: Subscription)")
plt.show()

# 4. Advanced Feature Engineering
def feature_engineering(df):
    # Create new features
    df['balance_to_age'] = df['balance'] / (df['age'] + 1)
    df['contact_frequency'] = df['campaign'] / (df['duration'] + 1)
    df['previous_contact_ratio'] = df['previous'] / (df['pdays'] + 100)  # +100 to avoid div by 0
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 5. Improved Encoding
cat_cols = train.select_dtypes(include='object').columns.tolist()
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = ord_enc.fit_transform(train[cat_cols])
test[cat_cols] = ord_enc.transform(test[cat_cols])

# 6. Feature Selection
X = train.drop(columns=['id', 'y'])
y = train['y']

# Calculate mutual information
mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi}).sort_values('mi_score', ascending=False)

# Select top features
selected_features = mi_df[mi_df['mi_score'] > 0.01]['feature'].tolist()
X = X[selected_features]
test = test[['id'] + selected_features]

# 7. Cross-Validation Setup
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = []
oof_preds = np.zeros(len(train))
feature_importance = pd.DataFrame()

# 8. Enhanced Model Training
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,  # Lower learning rate for better generalization
    'num_leaves': 63,      # Increased complexity
    'max_depth': 7,        # Constrained to prevent overfitting
    'min_child_samples': 20,
    'reg_alpha': 0.1,      # L1 regularization
    'reg_lambda': 0.1,     # L2 regularization
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(200)
        ]
    )
    
    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds.append(model.predict(test[selected_features]))
    
    # Feature importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = selected_features
    fold_importance["importance"] = model.feature_importance(importance_type='gain')
    fold_importance["fold"] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    # Fold evaluation
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

# 9. Model Evaluation
print("\nOverall Model Performance:")
print(f"OOF AUC: {roc_auc_score(y, oof_preds):.5f}")

# Feature Importance
mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importance.values, y=mean_importance.index)
plt.title("Feature Importance (Gain)")
plt.show()

# 10. Generate Predictions
test_pred = np.mean(test_preds, axis=0)
submission = pd.DataFrame({'id': test['id'], 'y': test_pred})
submission.to_csv("submission_enhanced.csv", index=False)
print("\nEnhanced submission file created successfully!")

