# Load these extensions to accelerate sklearn and pandas 
import warnings
warnings.filterwarnings('ignore')

%load_ext cuml.accel
%load_ext cudf.pandas


import numpy as np 
import pandas as pd 
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

from xgboost import XGBClassifier




train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

datasert_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

testID = test_df['id']
train_df = train_df.merge(datasert_df, how='left', on=merge_cols).drop(["id"],axis=1)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols).drop(["id"],axis=1)



# Numerical feature correlation
numeric_df = train_df.select_dtypes(include='number')
numeric_df.corr()


def groupwise_impute_train_test(train, test, group_col, target_col, q=4):
    """
    Perform quantile-based group-wise imputation on both train and test sets.
    - Binning and group medians are computed from train only (to avoid leakage).
    
    Parameters:
    - train: pd.DataFrame (with missing values)
    - test: pd.DataFrame (with missing values)
    - group_col: str - column used to group rows (e.g., 'Social_event_attendance')
    - target_col: str - column to impute (e.g., 'Time_spent_Alone')
    - q: int - number of quantile bins (default=4 for quartiles)
    
    Returns:
    - train, test: DataFrames with imputed `target_col`
    """

    # 1. Compute bin edges from train
    bin_edges = pd.qcut(train[group_col], q=q, retbins=True, duplicates='drop')[1]
    bin_labels = [f'Q{i+1}' for i in range(len(bin_edges) - 1)]

    # 2. Apply bins to both train and test
    train_bin_col = f'{group_col}_bin'
    test_bin_col = f'{group_col}_bin'
    
    train[train_bin_col] = pd.cut(train[group_col], bins=bin_edges, labels=bin_labels, include_lowest=True)
    test[test_bin_col] = pd.cut(test[group_col], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # 3. Compute medians in train
    group_medians = train.groupby(train_bin_col)[target_col].median()

    # 4. Impute missing values in train
    train[target_col] = train.apply(
        lambda row: group_medians.get(row[train_bin_col], np.nan) if pd.isnull(row[target_col]) else row[target_col],
        axis=1
    )

    # 5. Impute missing values in test
    test[target_col] = test.apply(
        lambda row: group_medians.get(row[test_bin_col], np.nan) if pd.isnull(row[target_col]) else row[target_col],
        axis=1
    )

    # 6. Drop temp columns
    train.drop(columns=[train_bin_col], inplace=True)
    test.drop(columns=[test_bin_col], inplace=True)

    return train, test



# Define the list of (group_source_col, target_col) pairs
group_target_pairs = [
    ('Social_event_attendance', 'Time_spent_Alone'),
    ('Going_outside', 'Time_spent_Alone'),
    ('Going_outside', 'Social_event_attendance'),
    ('Friends_circle_size', 'Social_event_attendance'),
    ('Post_frequency', 'Social_event_attendance'),
    ('Social_event_attendance', 'Going_outside'),
    ('Post_frequency', 'Friends_circle_size'),
    ('Going_outside', 'Friends_circle_size'),
    ('Friends_circle_size', 'Post_frequency')
]

# Apply group-wise imputation for each pair
for group_col, target_col in group_target_pairs:
    train_df,test_df = groupwise_impute_train_test(train_df, test_df, group_col, target_col, q=4)


print(train_df.info())
print(test_df.info())


train_df.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)
test_df.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)

print(train_df.info())
print(test_df.info())


train_df = pd.get_dummies(train_df, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])
test_df = pd.get_dummies(test_df, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])


X = train_df.drop(['Personality'],axis=1)
y = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.15, random_state=42, stratify=y)


xgb_params = {
    'n_estimators': 10000, 'objective': 'binary:logistic',  'eval_metric': 'error', 'eta': 0.05,
    'max_depth': 6, 'min_child_weight': 5, 'subsample': 0.85, 'colsample_bytree': 0.2,'lambda': 1,
    'alpha': 0.5, 'nthread': -1,
}


%%time
cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=234)
models = []
scores = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = XGBClassifier(**xgb_params, random_state=432)
    model.fit(X_tr, y_tr)

    # Predict and score
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    models.append(model)
    scores.append({'f1': f1, 'roc_auc': auc})

# Find the best fold
best_idx = max(range(len(scores)), key=lambda i: scores[i]['f1'])  # or 'roc_auc'
best_model = models[best_idx]
best_params = best_model.get_params()

print("Best F1 score:", scores[best_idx]['f1'])
print("Best model params:", best_params)



# Convert scores list to DataFrame
score_df = pd.DataFrame(scores)

# Melt to long format for Seaborn
score_long = score_df.melt(var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=score_long, x='Metric', y='Score', palette='Set2')
sns.stripplot(data=score_long, x='Metric', y='Score', color='black', alpha=0.5, jitter=True)

plt.title('Cross-Validation Score Distribution')
plt.ylabel('Score')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



final_model = XGBClassifier(
    **best_params,
)
final_model.fit(X_train,y_train)

y_val_predict=final_model.predict(X_val)

f1=f1_score(y_val, y_val_predict)
roc_auc=roc_auc_score(y_val, y_val_predict)
print(f"roc_auc:{roc_auc:0.4f} ")
print(f"f1:{f1:0.4f} ")


test_prediction=final_model.predict(test_df)

# Create submission
submission = pd.DataFrame({
    'id': testID,
    'Personality': test_prediction
})
submission.set_index('id', inplace=True)
print(submission.head())
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})

submission.to_csv('submission.csv')
print("Submitted successfully with XGBoost")

