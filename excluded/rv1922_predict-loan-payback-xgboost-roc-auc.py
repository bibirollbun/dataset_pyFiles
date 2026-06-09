import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


train.head()


TARGET = 'loan_paid_back'
CATS = ['gender', 'marital_status', 'education_level', 
        'employment_status', 'loan_purpose', 'grade_subgrade']


BASE = [col for col in train.columns if col not in ['id', TARGET]]
ORIG = []


for col in BASE:
    # MEAN encoding from Original Data
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(f'{len(ORIG)} Orig Features Created!!')
FEATURES = BASE + ORIG
print(f'{len(FEATURES)} Total Features.')


train.head()


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


for col in CATS:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cuda', 
    'enable_categorical': True,
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

print("\nStarting Training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'--- Fold {fold}/{N_SPLITS} AUC: {fold_score:.5f} ---')
    
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'\n==================================')
print(f'Overall OOF AUC: {overall_auc:.5f}')
print(f'==================================')


plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve (OOF)
plt.subplot(1, 2, 1)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.plot(fpr, tpr, color='blue', label=f'Overall AUC = {overall_auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (OOF Predictions)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 2: Histogram of Test Predictions
plt.subplot(1, 2, 2)
sns.histplot(test_preds, bins=50, kde=True, color='green')
plt.title('Distribution of Test Predictions')
plt.xlabel('Predicted Probability (loan_paid_back)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_preds  # Added quotes here
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved successfully!")


submission.describe()

