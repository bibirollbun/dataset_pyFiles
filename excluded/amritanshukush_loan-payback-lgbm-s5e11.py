import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


samp_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print("--------TRAIN DATA---------")
display(train_data.head(3))

print('\n')
print("\n--------TEST DATA----------")
display(test_data.head(3))

print('\n')
print("\n---------SAMPLE SUBMISSION---------")
display(samp_sub.head(3))


print('Train size : ' , train_data.shape)
print('Test size : ' , test_data.shape)


display(train_data.describe().T)
train_data.info()
train_data.isnull().sum()


counts = train_data['loan_paid_back'].value_counts()

plt.figure(figsize=(8,4))
sns.countplot(data=train_data, x='loan_paid_back', palette=['red' , 'green'])
plt.title('Distribution of Loan Paid Back')
plt.show()


num_cols = train_data.select_dtypes(include=['int' , 'float']).columns.drop('id')

plt.figure(figsize=(10,8))
corr_matrix = train_data[num_cols].corr()

sns.heatmap(corr_matrix , annot = True, cmap='viridis' , fmt = ".2f")
plt.title("Correlation between Numerical features")
plt.show()


CAT_COLS = train_data.select_dtypes(include=['object']).columns.tolist()
CAT_COLS


COLS = train_data.columns
for col in COLS:
    print(f"{col} : {train_data[col].unique()}")


for col in CAT_COLS:
    print(f"{col} : {train_data[col].unique()}")


ENCODE_COLS = ['gender',
 'marital_status',
 'education_level',
 'employment_status',
 'loan_purpose',
 'grade_subgrade']

train_data = pd.get_dummies(data = train_data , columns = ENCODE_COLS , dtype=int)
train_data.columns


test_data = pd.get_dummies(data = test_data , columns = ENCODE_COLS , dtype=int)


SCALE_COLS = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
scaler = MinMaxScaler()
train_data[SCALE_COLS] = scaler.fit_transform(train_data[SCALE_COLS])
train_data.head()


test_data[SCALE_COLS] = scaler.fit_transform(test_data[SCALE_COLS])


TARGET = 'loan_paid_back'


train_x = train_data.drop([TARGET , 'id'] , axis=1)
train_y = train_data[TARGET]

test_x = test_data.drop('id' , axis=1)


train_x.head()


test_x.head()


best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 128,
    'max_depth': 10,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    'max_bin': 255,           # helps with speed & stability
    'min_data_in_bin': 3
}


N_SPLITS = 5
lgb_models = []
lgb_scores = []
oof_preds = np.zeros(len(train_x))  # Out-of-fold predictions (probabilities)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("Training 5-fold LightGBM Binary Classifier...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
    print(f"Fold {fold + 1}/5")

    X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
    y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]

    # Create model with best params
    model = lgb.LGBMClassifier(**best_params)

    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ]
    )

    # Predict probabilities (NOT class labels!)
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred_proba

    # Calculate ROC AUC
    fold_auc = roc_auc_score(y_val, val_pred_proba)
    print(f"→ Fold {fold + 1} AUC: {fold_auc:.5f}")

    lgb_models.append(model)
    lgb_scores.append(fold_auc)

print("\n" + "="*50)
print(f"Mean CV ROC AUC: {np.mean(lgb_scores):.5f} ± {np.std(lgb_scores):.5f}")
print(f"Best single fold: {max(lgb_scores):.5f}")
print("="*50)


# Deploying trained model on test set
lgb_test_preds = sum(lgb_model.predict_proba(test_x)[:,1] for lgb_model in lgb_models) / len(lgb_models)


submission = pd.DataFrame({
    'id': test_data['id'],     # or .astype('float64') 
    'loan_paid_back': lgb_test_preds
})
submission.to_csv('submission.csv', index=False)
display(submission.head())


all_importances = np.array([model.feature_importances_ for model in lgb_models])
avg_importance = np.mean(all_importances, axis=0)

feature_importance_df = pd.DataFrame({
    'feature': train_x.columns,
    'importance': avg_importance
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'][:15], feature_importance_df['importance'][:15], color='skyblue')
plt.xlabel('Average Feature Importance (Gain)')
plt.ylabel('Feature')
plt.title('LightGBM Average Feature Importance (5-Fold Ensemble)')
plt.gca().invert_yaxis()   
plt.tight_layout()
plt.show()

