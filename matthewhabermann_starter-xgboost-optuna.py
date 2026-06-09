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



FRAC = 1 # Reduce this to troubleshoot and debug
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_train = df_train.sample(frac=FRAC, random_state=42).reset_index(drop=True)
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_train.set_index("id", inplace = True)

df_train.drop_duplicates(inplace = True)
df_train.head()


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import itertools
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector


class CrossFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = ['Calories', 'Sex']
        self.exclude_columns = exclude_columns
        self.cross_feature_names_ = []

    def fit(self, X, y=None):
        self.columns_ = X.columns[~X.columns.isin(self.exclude_columns)]
        return self

    def transform(self, X):
        X = X.copy()
        for col1, col2 in itertools.combinations(self.columns_, 2):
            diff_col = f'{col1}_minus_{col2}'
            prod_col = f'{col1}_times_{col2}'
            X[diff_col] = X[col1] - X[col2]
            X[prod_col] = X[col1] * X[col2]
            self.cross_feature_names_.extend([diff_col, prod_col])
        return X


class SexMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column='Sex', mapping=None):
        if mapping is None:
            mapping = {'male': 0, 'female': 1}
        self.column = column
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].map(self.mapping)
        return X


numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')) # It seems I don't need this, but it can't hurt. 
])


# --- Full Pipeline ---
preprocessor = Pipeline([
    ('sex_mapper', SexMapper()),
    ('cross_stats', CrossFeatureCreator()),
    ('column_transformer', ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, make_column_selector(dtype_include=['number']))
        ],
        remainder='passthrough'
    ))
])



from sklearn.linear_model import Ridge, LinearRegression, Lasso
import optuna
from sklearn.metrics import mean_squared_error
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score

target_column = "Calories"
y = df_train[target_column]
df_train = df_train.drop("Calories", axis = 1)





from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb


seed1 = 42
cv = KFold(n_splits=3, shuffle=True, random_state=seed1)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmsle',
    'device': 'cuda',
    'seed': seed1
}
# Hyperparameter tune using Optuna. This will take >1 hr on the Kaggle GPU.

def objective(trial):
    tuned_params = {
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
    }
    full_params = params.copy()
    full_params.update(tuned_params)
    fold_scores = []
    for train_idx, valid_idx in cv.split(df_train):
        X_train, y_train = df_train.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = df_train.iloc[valid_idx], y.iloc[valid_idx]

        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)

        dtrain = xgb.DMatrix(X_train_processed, label=y_train.values)
        dvalid = xgb.DMatrix(X_valid_processed, label=y_valid.values)

        model = xgb.train(
            full_params,
            dtrain,
            num_boost_round=10000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        preds = model.predict(dvalid)
        # Particularly early on in hyperparamter tuning, it is possible that the predictions are 
        # very bad, or even negative. Negative predictions in particular are bad, since 
        # you can't take their log. The following incentivises the algorithm to find better
        # parameters by setting the error to be infinite if there is a negative. 
        try:
            score = np.sqrt(mean_squared_log_error(y_valid, preds))
        except ValueError:
            score = float('inf')
            
        fold_scores.append(score)
    return np.mean(fold_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("âœ… Best RMSLE:", study.best_value)
print("ðŸŽ¯ Best Parameters:", study.best_params)


params.update(study.best_params)



final_preds = np.zeros(df_test.shape[0])

kf = KFold(n_splits=10, shuffle=True, random_state=seed1)

# The following essentially trains 10 models with the parameters found above, each with 1/10th of the data.
# Each model then makes a prediction, and the results are averaged. 

for idx_train, idx_valid in kf.split(df_train):
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
    X_test = df_test[df_train.columns].copy()
    
    
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)
    

    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dvalid = xgb.DMatrix(X_valid_processed, label=y_valid)
    dtest = xgb.DMatrix(X_test_processed)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=30,
        verbose_eval=500
    )

    final_preds += model.predict(dtest)

final_preds /= 10


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission['Calories'] = final_preds
submission.to_csv("submission.csv", index=False)
print("ðŸš€ Submission saved.")
submission.head()




