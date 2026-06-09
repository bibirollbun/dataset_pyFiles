# ===================== Library Imports =====================
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# ===================== Load Data =====================
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
train = train.drop_duplicates().reset_index(drop=True)
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

numerical = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            df[f"{features[i]}_x_{features[j]}"] = df[features[i]] * df[features[j]]
    return df


# ===================== Model 1 =====================
def predict_model_1(train):
    df = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    test_copy = test.copy()
    test_copy['Sex'] = le.transform(test_copy['Sex'])

    df = add_cross_terms(df, numerical)
    test_copy = add_cross_terms(test_copy, numerical)

    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    test_copy['BMI'] = test_copy['Weight'] / (test_copy['Height'] / 100) ** 2
    test_copy['Intensity'] = test_copy['Heart_Rate'] / test_copy['Duration']

    X = df.drop(columns=['Calories'])
    y = np.log1p(df['Calories'])
    X_test = test_copy[X.columns]

    bins = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(df[['Duration']]).astype(int).flatten()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))

    for train_idx, val_idx in skf.split(X, duration_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cat = CatBoostRegressor(verbose=0, random_state=42)
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        cat_preds += cat.predict(X_test) / skf.n_splits

        xgb = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=10,
                           subsample=0.9, colsample_bytree=0.7, gamma=0.01,
                           max_delta_step=2, tree_method="hist", enable_categorical=True,
                           early_stopping_rounds=100, eval_metric="rmse", verbosity=0,
                           random_state=42)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        xgb_preds += xgb.predict(X_test) / skf.n_splits

    final_preds = 0.5 * cat_preds + 0.5 * xgb_preds
    return np.clip(np.expm1(final_preds), 1, 314)



# ===================== Model 2 =====================
def predict_model_2(train):
    df = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
    test_copy = test.copy()

    for d in [df, test_copy]:
        d['Sex_Reversed'] = 1 - d['Sex']
        for dur in d['Duration'].unique():
            d[f'Heart_Rate_Duration_{int(dur)}'] = np.where(d['Duration'] == dur, d['Heart_Rate'], np.nan)
            d[f'Body_Temp_Duration_{int(dur)}'] = np.where(d['Duration'] == dur, d['Body_Temp'], np.nan)
        for age in d['Age'].unique():
            d[f'Heart_Rate_Age_{int(age)}'] = np.where(d['Age'] == age, d['Heart_Rate'], np.nan)
            d[f'Body_Temp_Age_{int(age)}'] = np.where(d['Age'] == age, d['Body_Temp'], np.nan)
        for f1 in ['Duration', 'Heart_Rate', 'Body_Temp']:
            for f2 in ['Sex', 'Sex_Reversed']:
                d[f'{f1}_x_{f2}'] = d[f1] * d[f2]
        d.drop(columns=['Sex_Reversed'], inplace=True)

    X = df.drop(columns=['Calories'])
    y = np.log1p(df['Calories'])
    test_copy = test_copy[X.columns]

    X['Sex'] = X['Sex'].astype('category')
    test_copy['Sex'] = test_copy['Sex'].astype('category')

    bins = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(df[['Duration']]).astype(int).flatten()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_test = np.zeros(len(test_copy))
    for train_idx, val_idx in skf.split(X, duration_bins):
        model = CatBoostRegressor(iterations=3500, learning_rate=0.02, depth=12,
                                  loss_function='RMSE', l2_leaf_reg=3,
                                  random_seed=42, eval_metric='RMSE',
                                  early_stopping_rounds=200, cat_features=['Sex'],
                                  verbose=0, task_type='GPU')
        model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        cat_test += np.expm1(model.predict(test_copy)) / skf.n_splits

    xgb_test = np.zeros(len(test_copy))
    X_xgb = X.copy()
    test_xgb = test_copy.copy()
    X_xgb['Sex'] = X_xgb['Sex'].astype(int)
    test_xgb['Sex'] = test_xgb['Sex'].astype(int)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_xgb):
        model = XGBRegressor(max_depth=9, colsample_bytree=0.7, subsample=0.9,
                             n_estimators=3000, learning_rate=0.01, gamma=0.01,
                             max_delta_step=2, eval_metric='rmse',
                             enable_categorical=False, random_state=42,
                             early_stopping_rounds=100, tree_method="hist", device="cuda")
        model.fit(X_xgb.iloc[train_idx], y.iloc[train_idx], eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])], verbose=0)
        xgb_test += np.expm1(model.predict(test_xgb)) / kf.n_splits

    return np.clip((cat_test * 0.5 + xgb_test * 0.5), 1, 314)


# ===================== Model 3 =====================
def predict_model_3(train):
    train = add_cross_terms(train, numerical)
    test_copy = add_cross_terms(test.copy(), numerical)
    train['Sex'] = train['Sex'].astype('category')
    test_copy['Sex'] = test_copy['Sex'].astype('category')

    X = train.drop(columns=['id', 'Calories'])
    y = np.log1p(train['Calories'])
    X_test = test_copy.drop(columns=['id'])

    duration_bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile') \
                      .fit_transform(train[['Duration']]).astype(int).flatten()
    skf = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)

    pred_cb = np.zeros(len(test_copy))
    pred_xgb = np.zeros(len(test_copy))

    for tr_idx, val_idx in skf.split(X, duration_bins):
        model_cb = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=10,
                                     l2_leaf_reg=3, loss_function='RMSE', eval_metric='RMSE',
                                     early_stopping_rounds=100, verbose=0,
                                     random_state=42, task_type="GPU",
                                     cat_features=[X.columns.get_loc("Sex")])
        model_cb.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
        pred_cb += model_cb.predict(X_test)

        model_xgb = XGBRegressor(max_depth=10, colsample_bytree=0.75, subsample=0.9,
                                 n_estimators=2000, learning_rate=0.02, gamma=0.01,
                                 max_delta_step=2, early_stopping_rounds=100,
                                 eval_metric="rmse", enable_categorical=True,
                                 tree_method="hist", device="cuda")
        model_xgb.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=0)
        pred_xgb += model_xgb.predict(X_test)

    pred_cb /= 50
    pred_xgb /= 50
    final_log = 0.3 * pred_cb + 0.7 * pred_xgb
    final = np.expm1(final_log)
    return np.clip(final, 1, 314)


# ===================== Final Ensemble (Weighted) =====================
def load_or_predict():
    if all(os.path.exists(f"pred{i}.npy") for i in range(1, 4)):
        print("âœ… Loading cached predictions...")
        return [np.load(f"pred{i}.npy") for i in range(1, 4)]
    else:
        print("ðŸš€ Running models and saving predictions...")
        pred1 = predict_model_1(train.copy()); np.save("pred1.npy", pred1)
        pred2 = predict_model_2(train.copy()); np.save("pred2.npy", pred2)
        pred3 = predict_model_3(train.copy()); np.save("pred3.npy", pred3)
        return [pred1, pred2, pred3]

pred1, pred2, pred3 = load_or_predict()
final_preds = 0.2 * pred1 + 0.4 * pred2 + 0.4 * pred3
submission['Calories'] = np.clip(final_preds, 1, 314)
submission.to_csv("submission_blend.csv", index=False)
print("ðŸŽ¯ submission_blend.csv saved with adaptive weighted ensemble âœ…")

