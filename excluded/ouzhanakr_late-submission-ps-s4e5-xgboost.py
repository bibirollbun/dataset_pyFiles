# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import optuna
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e5/sample_submission.csv')


train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


train.head()


train.isnull().sum()


def cross_validation(
    model,
    train,
    test=None,
    target="FloodProbability",
    features=None,
    n_splits=5,
    shuffle=True,
    random_state=1,
    single_fold=True,
    n_repeats=1,
    test_predict=False
):

    if features is None:
        features = [c for c in train.columns if c != target]

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof = np.full(len(train), np.nan, dtype=float)
    scores = []

    t0 = time.time()
    for fold, (i_train, i_val) in enumerate(kf.split(train)):
        X_train = train.iloc[i_train][features]
        y_train = train.iloc[i_train][target]
        X_val = train.iloc[i_val][features]
        y_val = train.iloc[i_val][target]

        y_pred = np.zeros(len(i_val), dtype=float)
        for s in range(n_repeats):
            m = clone(model)
            m.fit(X_train, y_train)
            y_pred += m.predict(X_val)
        y_pred /= n_repeats

        score = r2_score(y_val, y_pred)
        print(f"# Fold {fold}: R2={score:.5f}")
        scores.append(score)
        oof[i_val] = y_pred

        if single_fold:
            break

    r2_mean = float(np.mean(scores))
    elapsed_min = int(round((time.time() - t0) / 60))
    print(f"# Overall: {r2_mean:.5f} (single_fold={single_fold})  {elapsed_min} min")

    result = {"r2_mean": r2_mean, "scores": scores, "oof": oof}

    if test_predict and test is not None:
        m = clone(model)
        m.fit(train[features], train[target])
        test_pred = m.predict(test[features])
        result["test"] = test_pred

    return result



class FeatureEngineering(BaseEstimator, TransformerMixin):


    def __init__(self, columns=None, summaries=True, sort_values=True):
            self.columns = columns
            self.summaries = summaries
            self.sort_values = sort_values
            self.output_columns_ = None

    def fit(self,X,y=None):
        if self.columns is None:
            if isinstance(X, pd.DataFrame):
                self.columns = list(X.columns)
            else:
                raise ValueError("If columns=None, X must be a DataFrame.")

        out_cols = []
        if self.summaries:
            out_cols += ["fsum", "fmedian", "fstd"]
        if self.sort_values:
            out_cols += [f"sort_{i}" for i in range(len(self.columns))]
        self.output_columns_ = out_cols
        
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        mat = X[self.columns].to_numpy()
        parts = []

        if self.summaries:
            fsum = np.sum(mat, axis=1)
            fmedian = np.median(mat, axis=1)
            fstd = np.std(mat, axis=1)
            parts.append(np.c_[fsum, fmedian, fstd])

        if self.sort_values:
            sorted_vals = np.sort(mat, axis=1)  
            parts.append(sorted_vals)

        if len(parts) == 0:
            return X[self.columns].to_numpy()

        return np.concatenate(parts, axis=1)
        
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_, dtype=object)



target = 'FloodProbability'
features = [i for i in train.columns if i != target]


def build_pipeline_trial(trial, features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("pas", "passthrough", features),
            ("fe",  FeatureEngineering(columns=features, summaries=True, sort_values=True), features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    xgb = XGBRegressor(
        n_estimators     = trial.suggest_int("n_estimators", 600, 1600),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth        = trial.suggest_int("max_depth", 5, 10),
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 20.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        max_bin          = trial.suggest_categorical("max_bin", [256, 512, 1024, 2048]),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1

    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb',xgb),
    ])

    return pipeline




def build_pipeline_from_params(params, features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("pas", "passthrough", features),
            ("fe",  FeatureEngineering(columns=features, summaries=True, sort_values=True), features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    xgb = XGBRegressor(
        n_estimators     = params["n_estimators"],
        learning_rate    = params["learning_rate"],
        max_depth        = params["max_depth"],
        min_child_weight = params["min_child_weight"],
        subsample        = params["subsample"],
        colsample_bytree = params["colsample_bytree"],
        reg_lambda       = params["reg_lambda"],
        reg_alpha        = params["reg_alpha"],
        gamma            = params["gamma"],
        max_bin          = params["max_bin"],
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("xgb", xgb),
    ])
    return pipeline


def objective(trial):
    pipeline = build_pipeline_trial(trial, features)
    result = cross_validation(
        model=pipeline,
        train=train,
        test=None,
        target=target,        
        features=features,    
        n_splits=5,
        shuffle=True,
        random_state=42,
        single_fold=False,    
        n_repeats=1,          
        test_predict=False    
    )


    return result['r2_mean']


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_fe_pipeline_optuna")
study.optimize(objective, n_trials=10, show_progress_bar=True)

print("Best R2:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



best_pipeline = build_pipeline_from_params(study.best_params, features)

final_result = cross_validation(
    model=best_pipeline,
    train=train,
    test=test,
    target=target,         
    features=features,     
    n_splits=5,
    shuffle=True,
    random_state=42,
    single_fold=False,
    n_repeats=1,
    test_predict=True
)




sub = pd.read_csv('/kaggle/input/playground-series-s4e5/sample_submission.csv')  
submission = pd.DataFrame({
    'id': sub['id'],
    'FloodProbability': final_result['test']
})
submission.to_csv('submission.csv', index=False, float_format='%.6f')



submission




