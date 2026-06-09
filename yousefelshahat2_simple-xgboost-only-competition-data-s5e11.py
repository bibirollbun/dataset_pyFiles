pip install --upgrade xgboost scikit-learn


import pandas as pd
import numpy as np
import cupy as cp
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    binned = []
    
    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                    binned.append(f"{col}_bin{q}")
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0
                    binned.append(f"{col}_bin{q}")

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test, binned


def target_encoding(train, predict, n_splits=10):
    """
    The function turns categorical columns into numbers by replacing
    each category with its average target value, using K-Folds to avoid
    leaking information from the training data.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    for col in cat + add:
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train,train[target]):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply global mean mapping to prediction/test data ---
        global_mean = train.groupby(col)[target].mean()
        mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    predict = predict.copy()
    return train, predict


def target_encoding(train, predict, n_splits=10, smooth='auto'):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    for col in cat + add:
        # --- K-Fold Target Mean Encoding ---
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train, train[target]):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]

            counts = tr_fold.groupby(col)[target].count()
            means = tr_fold.groupby(col)[target].mean()

            # Determine smoothing factor
            m = smooth
            if smooth == 'auto':
                var_between = means.var()
                var_within = tr_fold.groupby(col)[target].var().mean()
                m = var_within / var_between if var_between > 0 else 0

            # Apply smoothing
            smooth_map = (counts * means + m * train[target].mean()) / (counts + m)
            mean_encoded[val_idx] = val_fold[col].map(smooth_map).fillna(train[target].mean())

        mean_features_train[f'mean_{col}'] = mean_encoded

        # --- Apply smoothed mapping to prediction/test data ---
        counts = train.groupby(col)[target].count()
        means = train.groupby(col)[target].mean()
        m = smooth
        if smooth == 'auto':
            var_between = means.var()
            var_within = train.groupby(col)[target].var().mean()
            m = var_within / var_between if var_between > 0 else 0

        smooth_map_global = (counts * means + m * train[target].mean()) / (counts + m)
        mean_features_test[f'mean_{col}'] = predict[col].map(smooth_map_global).fillna(train[target].mean())

    # --- Concatenate new features ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    return train, predict



# cols = df.drop(columns=[target,"id"]).columns.tolist()
# cat = [c for c in cols if df[c].dtype in ["object","category"]]
# gf, gf_test = target_encoding(df, df_test, 10)


class SafeTargetEncoder:
    """
    Leakage-safe target encoding for categorical or binned features.

    Parameters
    ----------
    cols : list of str
        Columns to target encode.
    target : str
        Name of the target column in train DataFrame.
    n_splits : int, default=5
        Number of CV folds for computing fold-wise encoding.
    smooth : float or 'auto', default='auto'
        Smoothing parameter for the mean encoding.
        If 'auto', uses empirical Bayes estimate.
    """
    def __init__(self, cols, target, n_splits=5, smooth='auto'):
        self.cols = cols
        self.target = target
        self.n_splits = n_splits
        self.smooth = smooth
        self.global_mean = None
        self.mappings_ = {}

    def fit(self, df):
        """Compute global statistics for unseen category handling."""
        self.global_mean = df[self.target].mean()
        return self

    def transform(self, df):
        """Apply encoding using learned mappings (for test data)."""
        df_enc = df.copy()
        for col in self.cols:
            mapping = self.mappings_.get(col, None)
            if mapping is None:
                raise ValueError(f"Column {col} not found in mappings. Did you fit_transform first?")
            df_enc[f'TE_{col}'] = df[col].map(mapping).fillna(self.global_mean)
        return df_enc

    def fit_transform(self, df, df_test=None):
        """Compute fold-wise encoding on train and apply to test."""
        df_enc = df.copy()
        mean_features_train = pd.DataFrame(index=df.index)
        mean_features_test = pd.DataFrame(index=df_test.index) if df_test is not None else None

        self.fit(df)  # compute global mean

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.cols:
            mean_encoded = np.zeros(len(df))

            for tr_idx, val_idx in kf.split(df,df[self.target]):
                tr_fold = df.iloc[tr_idx]
                val_fold = df.iloc[val_idx]

                counts = tr_fold.groupby(col)[self.target].count()
                means = tr_fold.groupby(col)[self.target].mean()

                # Smoothing
                m = self.smooth
                if m == 'auto':
                    var_between = means.var()
                    var_within = tr_fold.groupby(col)[self.target].var().mean()
                    m = var_within / var_between if var_between > 0 else 0

                smooth_mapping = (counts * means + m * self.global_mean) / (counts + m)
                mean_encoded[val_idx] = val_fold[col].map(smooth_mapping).fillna(self.global_mean)

            mean_features_train[f'TE_{col}'] = mean_encoded
            # Store global mapping for test
            counts = df.groupby(col)[self.target].count()
            means = df.groupby(col)[self.target].mean()
            m = self.smooth
            if m == 'auto':
                var_between = means.var()
                var_within = df.groupby(col)[self.target].var().mean()
                m = var_within / var_between if var_between > 0 else 0
            self.mappings_[col] = (counts * means + m * self.global_mean) / (counts + m)

            if df_test is not None:
                mean_features_test[f'TE_{col}'] = df_test[col].map(self.mappings_[col]).fillna(self.global_mean)

        df_enc = pd.concat([df_enc, mean_features_train], axis=1)
        if df_test is not None:
            df_test_enc = pd.concat([df_test, mean_features_test], axis=1)
            return df_enc, df_test_enc
        return df_enc



# Rounding the values
add = []
for c in ['annual_income', 'loan_amount']:
    for s, l in {'1s': 0, '10s': -1}.items():
        for g in [df, df_test]:
            g[f'{c}_ROUND_{s}'] = g[c].round(l).astype(int)
            add.append(f'{c}_ROUND_{s}')

# Specific feature engineering
for gf in [df, df_test]:
    gf['subgrade'] = gf['grade_subgrade'].str[1:].astype(int)
    gf['grade'] = gf['grade_subgrade'].str[0]
    gf['total_debt_burden'] = (gf['loan_amount'] * gf['interest_rate'] / 100) / (gf['annual_income'] + 1) 


cols = df.drop(columns=[target,"id"]).columns.tolist()
cat = [c for c in cols if df[c].dtype in ["object","category"]]
num = [c for c in cols if df[c].dtype not in ["object","category","bool"]]

# Creating new features based on the frequency of numerical features
df, df_test, binned = create_frequency_features(df, df_test)
te = SafeTargetEncoder(cols=cols, target=target, n_splits=10, smooth='auto')

df, df_test = te.fit_transform(df, df_test)
# df, df_test = target_encoding(df, df_test, 10)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq',"TE_total_debt_burden",'grade_subgrade',
    'annual_income_ROUND_1s', 'TE_annual_income', 'TE_gender', 'TE_marital_status', 'TE_education_level',
    'TE_employment_status', 'TE_grade_subgrade', 'TE_subgrade', 'interest_rate_freq', 'loan_amount_ROUND_1s_freq',
    'grade_freq', 'total_debt_burden_freq', 'annual_income_bin15', 'annual_income_ROUND_10s_bin5', 'total_debt_burden_bin5',
    'loan_amount','credit_score_bin15', 'loan_purpose_freq', 'total_debt_burden_bin15', 'debt_to_income_ratio_bin10', 'TE_subgrade_bin15',
    'TE_subgrade_bin10', 'employment_status_freq', 'total_debt_burden_bin10', 'loan_amount_ROUND_10s_bin15','TE_subgrade_bin15', 'TE_subgrade_bin10'
]

for i in remove:
    if i in df.columns:
        df, df_test = df.drop(columns = i), df_test.drop(columns = i)
df= df.drop(columns = ["id"])


print(f"Number of columns {len(df.columns.tolist())}")
print(df.columns.tolist())


df.isnull().sum()[lambda x: x>0]


df[target].value_counts()


dtrain = xgb.DMatrix(
    df.drop(columns = target),
    label=df[target],
    enable_categorical=True,
)

xgb_params = {
    'tree_method': 'hist', 'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic','random_state': 42,
    'min_child_weight': 89,"max_leaves":4,"reg_alpha":3.2,
    "reg_lambda":5,"eta":0.1,
}

cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=7,
    num_boost_round=20000,
    metrics='auc',
    verbose_eval=False,
    early_stopping_rounds=100,
)

print(cv_results.tail())

# Extract best boosting round
best_round = cv_results['test-auc-mean'].idxmax()
best_auc = cv_results['test-auc-mean'][best_round]
print(f"Best round: {best_round}, Best CV AUC: {best_auc:.7f}")


# Train final model
booster = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=best_round
)

# Get feature importances (weight: number of times a feature is used in splits)
importance = booster.get_score(importance_type='weight')

# All columns in your data
all_cols = set(df.drop(columns=target).columns)

# Columns not used in any split
unused_cols = all_cols - set(importance.keys())
print("Columns not used by XGBoost:", unused_cols)


k


# putting the n_estimator at the average early stopping point to avoid overfitting
last_round = len(cv_results)
xgb_params["n_estimators"] = last_round + 400


# Prepare training data
X_train = df.drop(columns=target)
y_train = df[target]

# Train XGBoost model
model = XGBClassifier(**xgb_params, enable_categorical=True)
model.fit(X_train, y_train,
    eval_set=[(X_train, y_train)],      # or (X_val, y_val) for real validation
    verbose=100)

# Predict on test set
pred = model.predict_proba(df_test.drop(columns = "id"))[:, 1]

# Prepare submission
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)




