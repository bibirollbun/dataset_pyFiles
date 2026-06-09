import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
#import pytabkit
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.metrics import *

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation,early_stopping
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings('ignore')
print('Done')


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = 'loan_paid_back'
print(df.shape)
print(df.columns.tolist())

df.head()


def create_frequency_features(df, df_test):
    df = df.copy()
    df_test = df_test.copy()

    freq_features_train = {}
    freq_features_test = {}
    bin_features_train = {}
    bin_features_test = {}

    for col in cols:

        # Frequency Encoding
        freq = df[col].value_counts()

        # train freq
        freq_features_train[f"{col}_freq"] = df[col].map(freq)

        # test freq (fix unseen categories)
        default_value = freq.mean() if len(freq) > 0 else 0
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(default_value)

        # Quantile Binning (numeric only)
        if col in num:
            for q in (5, 10, 15):

                try:
                    t_bins, edges = pd.qcut(df[col], q=q, labels=False,
                                            retbins=True, duplicates='drop')

                    bin_features_train[f"{col}_bin{q}"] = t_bins

                    # Use same edges for test
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(
                        df_test[col],
                        bins=edges,
                        labels=False,
                        include_lowest=True
                    )

                except Exception:
                    # If qcut fails (constant column, few unique values...)
                    bin_features_train[f"{col}_bin{q}"] = pd.Series(0, index=df.index)
                    bin_features_test[f"{col}_bin{q}"] = pd.Series(0, index=df_test.index)

    # Merge all new features
    df = pd.concat([df, pd.DataFrame(freq_features_train),
                        pd.DataFrame(bin_features_train)], axis=1)

    df_test = pd.concat([df_test, pd.DataFrame(freq_features_test),
                               pd.DataFrame(bin_features_test)], axis=1)

    return df, df_test



def target_encoding(train, predict, n_splits=8):
    train = train.copy()
    predict = predict.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = {}
    mean_features_test = {}

    # Compute global mean once
    target_global = train[target].mean()

    for col in cols:

        # K-FOLD TARGET ENCODING
        oof = np.zeros(len(train))

        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]

            # Mean for each category in this fold
            fold_map = tr_fold.groupby(col)[target].mean()

            # Map on validation fold
            oof[val_idx] = train[col].iloc[val_idx].map(fold_map).fillna(target_global)

        mean_features_train[f"mean_{col}"] = oof

        # Apply encoding to the prediction set
        global_map = train.groupby(col)[target].mean()
        mean_features_test[f"mean_{col}"] = (
            predict[col].map(global_map).fillna(target_global)
        )

    # ------------------------------------
    # Attach all new features at once
    # ------------------------------------
    train = pd.concat([train, pd.DataFrame(mean_features_train)], axis=1)
    predict = pd.concat([predict, pd.DataFrame(mean_features_test)], axis=1)

    return train, predict



# Extract grade + subgrade
for df_ in [df, df_test]:
    df_["subgrade"] = df_["grade_subgrade"].str[1:].astype(int)
    df_["grade"]    = df_["grade_subgrade"].str[0]

# Identify feature list
cols = (
    df.drop(columns=[target, "id"], errors="ignore")
      .columns
      .tolist()
)

# Categorical
cat = [c for c in cols if df[c].dtype in ("object", "category")]

# Numeric
num = [c for c in cols if df[c].dtype not in ("object", "category", "bool")]

#  Feature Engineering (Target + Frequency + Binning)
df, df_test = target_encoding(df, df_test, n_splits=10)
df, df_test = create_frequency_features(df, df_test)

# Convert categoricals properly
df[cat] = df[cat].astype("category")
df_test[cat] = df_test[cat].astype("category")

# Drop unwanted features safely
remove = [
    "education_level", "loan_purpose", "grade_subgrade",
    "interest_rate", "marital_status", "gender",
    "employment_status_freq", "credit_score_bin5",
    "loan_amount_bin5", "credit_score_freq",
    "mean_subgrade", "subgrade_bin15", "subgrade_bin10",
    "debt_to_income_ratio_bin5"
]

df.drop(columns=[c for c in remove if c in df.columns], inplace=True)
df_test.drop(columns=[c for c in remove if c in df_test.columns], inplace=True)

# Final cleaning
df.drop(columns="id", errors="ignore", inplace=True)
df.drop_duplicates(inplace=True)



X = df.drop(columns=[target])
y = df[target]

lgb_train = lgb.Dataset(X, label=y, free_raw_data=True)
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",  # or try 'dart' later
    "learning_rate": 0.03,
    "num_leaves": 80,
    "max_depth": 6,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 2,
    
    "feature_fraction": 0.85,
    "bagging_fraction": 0.9,

    # Regularization (important for stable CV)
    "reg_alpha": 0.2,
    "reg_lambda": 0.4,
    "min_split_gain": 0.01,
    "min_data_in_leaf": 40,

    "n_jobs": -1,
    "device": "gpu",
    "verbose": -1,
    "random_state": 42
}
cv_results = lgb.cv(
    params=lgb_params,
    train_set=lgb_train,
    num_boost_round=20000,
    nfold=10,
    stratified=True,
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period = 150)],
    seed=42
)

cv_df = pd.DataFrame(cv_results)
print(cv_df.tail())

best_round = len(cv_results['valid auc-mean'])
best_auc = cv_results['valid auc-mean'][-1]
print(f"Best round: {best_round}, Best CV AUC: {best_auc:.7f}")


lgb_params["n_estimators"] = best_round + 100
print(best_round)


# Prepare training data
X_train = df.drop(columns=target)
y_train = df[target]

# Train LGBM model
model = LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)

# Predict on test set
pred = model.predict_proba(df_test.drop(columns = "id"))[:, 1]

# Prepare submission
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)


sub.head()

