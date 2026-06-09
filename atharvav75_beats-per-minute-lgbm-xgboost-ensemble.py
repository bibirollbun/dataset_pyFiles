import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e9'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install lightgbm
!pip install xgboost


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import xgboost as xgb


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


def data_overview(df):
    summary = pd.DataFrame({
        "DataType": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "%Missing Value": (df.isnull().sum() / len(df)) * 100
    })
    return summary.reset_index().rename(columns={"index": "Features"})
    data_overview(df)


numeric_cols = df.drop('id', axis=1).select_dtypes(include=['int64', 'float64']).columns
n_cols = 4
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
bins = 30

colors = cm.tab20(np.linspace(0, 1, len(numeric_cols)))
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, (col, color) in enumerate(zip(numeric_cols, colors), 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df[col], bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.title(col)
    plt.tight_layout()
plt.show()


x = df.drop(["id", "BeatsPerMinute"], axis=1)
test = test_df.drop("id", axis=1)
y = df["BeatsPerMinute"]

skewed_cols = ["VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood"]
standardize_cols = ["AudioLoudness"]

preprocessor = ColumnTransformer(
    transformers=[
        ("skewed", RobustScaler(), skewed_cols),
        ("standardize", StandardScaler(), standardize_cols),
    ],
    remainder="passthrough"
)


X_transformed = preprocessor.fit_transform(x)
x_test = preprocessor.transform(test)

feature_names = preprocessor.get_feature_names_out()
X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=x.index)
x_test = pd.DataFrame(x_test, columns=feature_names, index=test.index)

# Add interaction features
X_transformed['inter_energy_rhythm'] = X_transformed['remainder__Energy'] * X_transformed['remainder__RhythmScore']
X_transformed['inter_mood_energy'] = X_transformed['remainder__MoodScore'] * X_transformed['remainder__Energy']
x_test['inter_energy_rhythm'] = x_test['remainder__Energy'] * x_test['remainder__RhythmScore']
x_test['inter_mood_energy'] = x_test['remainder__MoodScore'] * x_test['remainder__Energy']

# Add polynomial features
X_transformed['poly_duration'] = X_transformed['remainder__TrackDurationMs'] ** 2
X_transformed['poly_loudness'] = X_transformed['standardize__AudioLoudness'] ** 2
x_test['poly_duration'] = x_test['remainder__TrackDurationMs'] ** 2
x_test['poly_loudness'] = x_test['standardize__AudioLoudness'] ** 2


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {"lgb": np.zeros(len(X_transformed)), "xgb": np.zeros(len(X_transformed))}
test_preds = {"lgb": np.zeros((len(x_test), kf.get_n_splits())), "xgb": np.zeros((len(x_test), kf.get_n_splits()))}
scores = {"lgb": [], "xgb": []}


for fold, (train_idx, valid_idx) in enumerate(kf.split(X_transformed, y), 1):
    X_train, X_valid = X_transformed.iloc[train_idx], X_transformed.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(random_state=42, verbose=-1, device='gpu')
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'reg_lambda': [0, 10]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    preds_valid = best_model.predict(X_valid)
    oof_preds["lgb"][valid_idx] = preds_valid
    r2 = r2_score(y_valid, preds_valid)
    scores["lgb"].append(r2)
    print(f"Fold {fold}: R2 = {r2:.4f}, Best params: {grid.best_params_}")
    
    preds_test = best_model.predict(x_test)
    test_preds["lgb"][:, fold-1] = preds_test

print(f"CV R2: {np.mean(scores['lgb']):.4f} ± {np.std(scores['lgb']):.4f}\n")


for fold, (train_idx, valid_idx) in enumerate(kf.split(X_transformed, y), 1):
    X_train, X_valid = X_transformed.iloc[train_idx], X_transformed.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
    param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'reg_lambda': [0, 10]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    preds_valid = best_model.predict(X_valid)
    oof_preds["xgb"][valid_idx] = preds_valid
    r2 = r2_score(y_valid, preds_valid)
    scores["xgb"].append(r2)
    print(f"Fold {fold}: R2 = {r2:.4f}, Best params: {grid.best_params_}")
    
    preds_test = best_model.predict(x_test)
    test_preds["xgb"][:, fold-1] = preds_test

print(f"CV R2: {np.mean(scores['xgb']):.4f} ± {np.std(scores['xgb']):.4f}\n")


final_preds = 0.5 * test_preds["lgb"].mean(axis=1) + 0.5 * test_preds["xgb"].mean(axis=1)


submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": final_preds
})
submission.to_csv("submission.csv", index=False)

