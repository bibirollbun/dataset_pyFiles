# =============================================================================
# 0️⃣ IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from scipy.optimize import nnls
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# =============================================================================
# 1️⃣ HELPER: CROSS‐TERM GENERATOR
# =============================================================================
def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            a, b = features[i], features[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df

# =============================================================================
# 2️⃣ LOAD & BASE PREPROCESSING
# =============================================================================
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# remove exact duplicates
train.drop_duplicates(inplace=True)
train.reset_index(drop=True, inplace=True)

# encode Sex as numerical
for df in (train, test):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# =============================================================================
# 3️⃣ MODEL 1: Duration‐Aware Dual Boosting
# =============================================================================
def predict_model_1(train_df, test_df):
    grp = ['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
    df = train_df.groupby(grp)['Calories'].min().reset_index()
    test_copy = test_df.copy()

    # label‐encode Sex
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    test_copy['Sex'] = le.transform(test_copy['Sex'])

    # interaction & engineered features
    base_feats = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
    df = add_cross_terms(df, base_feats)
    test_copy = add_cross_terms(test_copy, base_feats)
    for d in (df, test_copy):
        d['BMI']       = d['Weight'] / (d['Height']/100)**2
        d['Intensity'] = d['Heart_Rate'] / d['Duration']

    X = df.drop(columns=['Calories'])
    y = np.log1p(df['Calories'])
    X_test = test_copy[X.columns]

    # stratify by binned Duration
    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    d_bins = bins.fit_transform(df[['Duration']]).astype(int).ravel()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))

    for tr, va in skf.split(X, d_bins):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # CatBoost on log-target
        cat = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False, early_stopping_rounds=50
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        cat_preds += np.expm1(cat.predict(X_test)) / skf.n_splits

        # XGBoost on log-target
        xgb = XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.7, gamma=0.01,
            tree_method='hist', random_state=42, verbosity=0
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                early_stopping_rounds=100, verbose=False)
        xgb_preds += np.expm1(xgb.predict(X_test)) / skf.n_splits

    # blend 50/50 and clip
    return np.clip(0.5*cat_preds + 0.5*xgb_preds, 1, 314)

# =============================================================================
# 4️⃣ MODEL 2: Conditional Interaction Ensemble
# =============================================================================
def predict_model_2(train_df, test_df):
    grp = ['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
    df = train_df.groupby(grp)['Calories'].min().reset_index()
    test_copy = test_df.copy()

    # add reversed Sex
    for d in (df, test_copy):
        d['Sex_Reversed'] = 1 - d['Sex']

    # conditional masking & interactions
    for d in (df, test_copy):
        for dur in d['Duration'].unique():
            mask = (d['Duration']==dur).astype(int)
            d[f'HR_if_Dur_{dur}'] = d['Heart_Rate'] * mask
            d[f'BT_if_Dur_{dur}'] = d['Body_Temp']  * mask
        for age in d['Age'].unique():
            mask = (d['Age']==age).astype(int)
            d[f'HR_if_Age_{age}'] = d['Heart_Rate'] * mask
            d[f'BT_if_Age_{age}'] = d['Body_Temp']  * mask
        for feat in ['Duration','Heart_Rate','Body_Temp']:
            d[f'{feat}_x_Sex']    = d[feat] * d['Sex']
            d[f'{feat}_x_SexRev'] = d[feat] * d['Sex_Reversed']
        d.drop(columns=['Sex_Reversed'], inplace=True)

    X = df.drop(columns=['Calories'])
    y = np.log1p(df['Calories'])
    X_test = test_copy[X.columns]

    # stratify by Duration
    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    d_bins = bins.fit_transform(df[['Duration']]).astype(int).ravel()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    cat_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))

    # CatBoost
    for tr, va in skf.split(X, d_bins):
        cat = CatBoostRegressor(
            iterations=1200, learning_rate=0.04, depth=8,
            random_seed=42, verbose=False, early_stopping_rounds=80
        )
        cat.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
        cat_preds += np.expm1(cat.predict(X_test)) / skf.n_splits

    # XGBoost
    for tr, va in kf.split(X):
        xgb = XGBRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=9,
            subsample=0.9, colsample_bytree=0.8,
            tree_method='hist', random_state=42, verbosity=0
        )
        xgb.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                early_stopping_rounds=100, verbose=False)
        xgb_preds += np.expm1(xgb.predict(X_test)) / kf.n_splits

    return np.clip(0.5*cat_preds + 0.5*xgb_preds, 1, 314)

# =============================================================================
# 5️⃣ MODEL 3: High‐Fold Log‐Domain Ensemble
# =============================================================================
def predict_model_3(train_df, test_df, folds=20):
    df = train_df.copy()
    feats = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Sex']
    df = add_cross_terms(df, feats)
    test_copy = add_cross_terms(test_df.copy(), feats)
    for d in (df, test_copy):
        d['BMI']      = d['Weight'] / (d['Height']/100)**2
        d['Workload'] = d['Duration'] * d['Heart_Rate']

    X = df.drop(columns=['Calories','id'])
    y = np.log1p(df['Calories'])
    X_test = test_copy[X.columns]

    skf = KFold(n_splits=folds, shuffle=True, random_state=42)
    cat_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))

    for tr, va in skf.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        cat = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False, early_stopping_rounds=30
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        cat_preds += np.expm1(cat.predict(X_test)) / folds

        xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=42, verbosity=0
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                early_stopping_rounds=20, verbose=False)
        xgb_preds += np.expm1(xgb.predict(X_test)) / folds

    return np.clip(0.3*cat_preds + 0.7*xgb_preds, 1, 314)

# =============================================================================
# 6️⃣ HOLD‐OUT BLEND WEIGHTS & FINAL SUBMISSION
# =============================================================================
# 6.1 split off 10% hold‐out
train_base, holdout = train_test_split(train, test_size=0.10, random_state=42)

# 6.2 get hold‐out predictions
p1_val = predict_model_1(train_base, holdout)
p2_val = predict_model_2(train_base, holdout)
p3_val = predict_model_3(train_base, holdout, folds=5)

# 6.3 learn NNLS weights
OOF_val = np.vstack([p1_val, p2_val, p3_val]).T
w, _ = nnls(OOF_val, holdout['Calories'])
print("Learned blend weights:", w)

# 6.4 predict on full test
p1_test = predict_model_1(train, test)
p2_test = predict_model_2(train, test)
p3_test = predict_model_3(train, test, folds=20)

final = w[0]*p1_test + w[1]*p2_test + w[2]*p3_test
final = np.clip(final, 1, 314)

# 6.5 save submission
submission['Calories'] = final
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved!")

