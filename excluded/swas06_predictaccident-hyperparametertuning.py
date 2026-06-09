# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.shape,df_test.shape


df_train.info()


df_train.head(3),df_test.head(3)


numerical_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")


features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


scaler = StandardScaler()
df_train[features] = scaler.fit_transform(df_train[features])


df_test[features] = scaler.fit_transform(df_test[features])


df_train.head(3)


categorical_columns = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
for col in categorical_columns:
    print(col, "-->", df_train[col].unique())


bool_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']


df_train[bool_features] = df_train[bool_features].astype(int)


df_test[bool_features] = df_test[bool_features].astype(int)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_array = encoder.fit_transform(df_train[cat_features]).toarray()


encoded_cols = encoder.get_feature_names_out(cat_features)


encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_train.index)


df_final = pd.concat([df_train.drop(columns=cat_features), encoded_df], axis=1)


df_train.head(3)


df_final.head(3)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_array = encoder.fit_transform(df_test[cat_features]).toarray()


encoded_cols = encoder.get_feature_names_out(cat_features)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_test.index)



# Drop original categorical columns
df_test_final = pd.concat([df_test.drop(columns=cat_features), encoded_df], axis=1)



df_test_final.head(3)


X =df_final.drop('accident_risk', axis='columns')
y = df_train['accident_risk']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, make_scorer


from xgboost import XGBRegressor
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    model = XGBRegressor(**params)

    # Cross-validation for robust results
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

    scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)

    mean_rmse = -scores.mean()  # since greater_is_better=False, scores are negative

    return mean_rmse


from xgboost import XGBRegressor

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)
print("Best XGBoost Params:", study_xgb.best_params)
print("Best RMSE:", study_xgb.best_value)


best_xgb = XGBRegressor(**study_xgb.best_params)
best_xgb.fit(X_train, y_train)


from xgboost import XGBRegressor
model = XGBRegressor(
    **study_xgb.best_params,
    use_label_encoder=False,   # only needed for classification
    eval_metric='mlogloss',
    random_state=42
)


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

cv = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_scores = []   # Store RMSE for each fold
all_preds = []     # Store predictions for test set from each fold

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"Training fold {fold}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model (XGBRegressor or ensemble)
    model.fit(X_train, y_train)

    # Predict for validation set
    y_val_pred = model.predict(X_val)

    # Predict for test set
    test_pred = model.predict(df_test_final)

    # Calculate RMSE for this fold
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_scores.append(rmse)

    # Save test predictions for later averaging
    all_preds.append(test_pred)

# --- Mean Cross-Validation RMSE ---
mean_rmse = np.mean(rmse_scores)

# --- Accuracy replaced with CV RMSE ---
cv_rmse = -cross_val_score(model, X, y, cv=5,
                           scoring='neg_root_mean_squared_error').mean()

# --- Results ---
print(f"\nRMSE per fold: {rmse_scores}")
print(f"Mean RMSE (manual CV): {mean_rmse:.4f}")
print(f"Cross-Validated RMSE (using cross_val_score): {cv_rmse:.4f}")



sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


final_preds = np.mean(all_preds, axis=0)

# --- Submission ---
submission = pd.DataFrame({
    'id': sample_submission.id,
    'y': final_preds
})
submission.to_csv('submission_ensemble.csv', index=False)
print(submission.head())

