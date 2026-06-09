pip install --upgrade xgboost scikit-learn


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


df_target = df[target]


# def create_frequency_features(df, df_test):
#     """
#     Add frequency and binning features efficiently.

#     - For each categorical column, create <col>_freq = how often each value appears in train data.
#     - For numeric columns, split values into 5, 10, 15 quantile bins.
#     """
#     # Pre-allocate DataFrames for new features to avoid fragmentation
#     freq_features_train = pd.DataFrame(index=df.index)
#     freq_features_test = pd.DataFrame(index=df_test.index)
#     bin_features_train = pd.DataFrame(index=df.index)
#     bin_features_test = pd.DataFrame(index=df_test.index)

#     for col in cols:
#         # --- Frequency encoding ---
#         freq = df[col].value_counts()
#         df[f"{col}_freq"] = df[col].map(freq)
#         freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

#         # --- Quantile binning for numeric columns ---
#         if col in num:
#             for q in [5, 10, 15]:
#                 try:
#                     train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
#                     bin_features_train[f"{col}_bin{q}"] = train_bins
#                     bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
#                 except Exception:
#                     bin_features_train[f"{col}_bin{q}"] = 0
#                     bin_features_test[f"{col}_bin{q}"] = 0

#     # Concatenate all new features at once
#     df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
#     df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

#     return df, df_test


# def target_encoding(train, predict, n_splits=5):
#     """
#     Add K-Fold target mean encoded features to train and predict datasets.
    
#     Parameters:
#     - train: training DataFrame
#     - predict: prediction/test DataFrame
#     - target: name of the target column
#     - n_splits: number of folds for K-Fold encoding
    
#     Returns:
#     - train and predict DataFrames with new mean encoded features
#     """
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     mean_features_train = pd.DataFrame(index=train.index)
#     mean_features_test = pd.DataFrame(index=predict.index)

#     for col in cols:
#         # --- K-Fold Target Mean Encoding ---
#         mean_encoded = np.zeros(len(train))
#         for tr_idx, val_idx in kf.split(train):
#             tr_fold = train.iloc[tr_idx]
#             val_fold = train.iloc[val_idx]
#             mean_map = tr_fold.groupby(col)[target].mean()
#             mean_encoded[val_idx] = val_fold[col].map(mean_map)

#         mean_features_train[f'mean_{col}'] = mean_encoded

#         # --- Apply global mean mapping to prediction/test data ---
#         global_mean = train.groupby(col)[target].mean()
#         mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

#     # --- Concatenate new features at once to avoid fragmentation ---
#     train = pd.concat([train, mean_features_train], axis=1)
#     predict = pd.concat([predict, mean_features_test], axis=1)

#     # Defragment
#     train = train.copy()
#     predict = predict.copy()
#     return train, predict


df['grade_subgrade']


# Specific feature engineering
df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
df_test['subgrade'] = df_test['grade_subgrade'].str[1:].astype(int)


# Identify feature
cols = df.drop(columns=[target,"id"]).columns.tolist()
cols


# Categorical features
cat = [c for c in cols if df[c].dtype in ["object","category"]]

# Numerical features
num = [c for c in cols if df[c].dtype not in ["object","category","bool"]]


df[num]


df[cat]


# Creating new features based on the frequency of numerical features
# df, df_test = target_encoding(df, df_test, 10)
# df, df_test = create_frequency_features(df, df_test)

# Preparing categorical features
#df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Dropping unnecessary columns
# remove = ["education_level","loan_purpose", "grade_subgrade", "interest_rate", "marital_status",
#           "gender", "employment_status_freq", "credit_score_bin5",  "loan_amount_bin5",
#           "debt_to_income_ratio_bin5"]
#df, df_test = df.drop(columns = remove), df_test.drop(columns = remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)


test_id = df_test['id']


df = df[num]
df_test = df_test[num]


print(df.columns.tolist())


df.isnull().sum()[lambda x: x>0]


df.isnull().sum().sum()


df.shape


df_test.shape


# def give_score():
dtrain = xgb.DMatrix(
    #df.drop(columns=[target]),
    df,
    label=df_target,
    enable_categorical=True,
)

xgb_params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'scale_pos_weight':1,
    'min_child_weight': 89,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    "gamma":0,
    "max_leaves":4,
    "reg_alpha":1.4,
    "reg_lambda":5.9,
    "scale_pos_weight":1,
}


cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=7,
    num_boost_round=20000,
    metrics='auc',
    verbose_eval=False,
    early_stopping_rounds=50,
)

print(cv_results.tail())

# Extract best boosting round
best_round = cv_results['test-auc-mean'].idxmax()
best_auc = cv_results['test-auc-mean'][best_round]
print(f"Best round: {best_round}, Best CV AUC: {best_auc:.7f}")


# putting the n_estimator at the average early stopping point to avoid overfitting
last_round = len(cv_results) - 1
xgb_params["n_estimators"] = last_round + 10


last_round


X_train.shape


X_train.head()


y_train


# Prepare training data
X_train = df
y_train = df_target

# Train XGBoost model
model = XGBClassifier(**xgb_params, enable_categorical=True)
model.fit(X_train, y_train)

# Predict on test set
pred = model.predict_proba(df_test)[:, 1]

# Prepare submission
sub = pd.DataFrame({
    "id": test_id,
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)


import matplotlib.pyplot as plt

# Plot the first tree (num_trees=0)
xgb.plot_tree(model, num_trees=0)
plt.rcParams['figure.figsize'] = [30, 30]  # Optional: make it larger
plt.show()


xgb.plot_tree(model, num_trees=1)
plt.rcParams['figure.figsize'] = [30, 30]  # Optional: make it larger
plt.show()


df_a = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


df_a['loan_paid_back'].mean()


mask = df_a['debt_to_income_ratio'] <0.179


df_a[mask]['loan_paid_back'].mean()


mask_cs = df_a['credit_score'] < 671


df_a[mask&mask_cs]['loan_paid_back'].mean()


df_a[mask&(~mask_cs)]['loan_paid_back'].mean()


df_a[~mask]['loan_paid_back'].mean()


df_a[~mask&mask_cs]['loan_paid_back'].mean()


df_a[mask]['loan_paid_back'].mean()


df_a[~mask]['loan_paid_back'].mean()


mask




