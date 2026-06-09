import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import FunctionTransformer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings

warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

experiments = pd.DataFrame(columns=["Name", "FeaturesAdded", "Score", "Deviation"])


def load_data():
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

    df_external_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
    df_external_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

    return df_train, df_external_train, df_test, df_external_test


def check_scores(pipeline, X, y):
    scores = -1 * cross_val_score(pipeline, X, y,
                                  cv=5,
                                  scoring='neg_root_mean_squared_error')
    print("RMSE score:\n", np.mean(scores), "Deviation:\n", np.std(scores))
    return np.mean(scores), np.std(scores)


%%time
default_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1
)
pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', default_model)
])

df_train, df_external_train, df_test, df_external_test = load_data()
X = df_train.copy()
y = X.pop("BeatsPerMinute")
X = X.drop(columns=["id"])

score, deviation = check_scores(pipeline, X, y)
experiments.loc[len(experiments)] = ["Default", "Basic Dataset", score, deviation]


%%time
X_full = pd.concat([df_train, df_external_train])
y_full = X_full.pop("BeatsPerMinute")
X_full = X_full.drop(columns=["id"])

def mutual_info_ranking(X_full, y_full):
    scores = mutual_info_regression(
        X, y,
        discrete_features=False,
        random_state=42
    )
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

ranking = mutual_info_ranking(X, y)
print(ranking)


X_full.head()


X_full.shape


%%time
# Explore a bit the different cols
plt.figure(figsize=(15,4))

# Plot each column
for i, col in enumerate(X_full.columns, 1):
    plt.subplot(3,4,i)
    sns.histplot(X_full[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


%%time
def feature_engineering(X: pd.DataFrame):
    Xf = X.copy()
    eps = 1e-6

    num_cols = Xf.select_dtypes(include=[np.number]).columns
    Xf[num_cols] = Xf[num_cols].fillna(Xf[num_cols].median())
    
    if {'VocalContent','AcousticQuality'}.issubset(Xf.columns):
        Xf['Vocal_Acoustic_Quot'] = Xf['VocalContent'] / (Xf['AcousticQuality'] + eps)
    if {'RhythmScore','Energy'}.issubset(Xf.columns):
        Xf['Rhythm_Energy_Ratio'] = Xf['RhythmScore'] / (Xf['Energy'] + eps)

    for var in ['TrackDurationMs','AudioLoudness','VocalContent','Energy','InstrumentalScore']:
        if var in Xf.columns and skew(Xf[var].dropna()) > 0.5:
            adj = -(Xf[var].min())+1 if Xf[var].min() < 0 else 0
            Xf[f'log_{var}'] = np.log1p(Xf[var] + adj)

    if 'TrackDurationMs' in Xf.columns:
        Xf['Duration_Cat'] = pd.qcut(Xf['TrackDurationMs'], q=10, labels=False, duplicates='drop')
    if 'Energy' in Xf.columns:
        Xf['Energy_Cat'] = pd.qcut(Xf['Energy'], q=5, labels=False, duplicates='drop')
    if 'AudioLoudness' in Xf.columns:
        Xf['Loudness_Cat'] = pd.qcut(Xf['AudioLoudness'], q=5, labels=False, duplicates='drop')
    if 'VocalContent' in Xf.columns:
        Xf['Vocal_Cat'] = pd.qcut(Xf['VocalContent'], q=5, labels=False, duplicates='drop')

    return Xf
    
feat_eng_transformer = FunctionTransformer(feature_engineering, validate=False)
X_full = feature_engineering(X_full)


%%time
ranking = mutual_info_ranking(X_full, y_full)
print(ranking)


%%time
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1
)

lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1
)

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_state=0,
    verbose=0
)

pipelines = {
    "xgb": Pipeline([
        ("features", feat_eng_transformer),
        ("model", xgb_model)
    ]),
    "lgbm": Pipeline([
        ("features", feat_eng_transformer),
        ("model", lgbm_model)
    ]),
    "cat": Pipeline([
        ("features", feat_eng_transformer),
        ("model", cat_model)
    ]),
}

for name, pipe in pipelines.items():
    print(f"Model: {name}")
    check_scores(pipe, X_full, y_full)


pip install optuna-integration[xgboost]


import optuna
from optuna.samplers import TPESampler

def optimize_xgb_regression(
    X, y,
    feat_eng_transformer,
    n_trials: int = 50,
    timeout: int | None = None,
    random_state: int = 42
):

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 16.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "random_state": random_state,
            "n_jobs": -1,
        }

        model = XGBRegressor(**params)

        pipeline = Pipeline([
            ("preprocessor", SimpleImputer(strategy="mean")),
            ("features", feat_eng_transformer),
            ("model", model),
        ])

        mean_rmse, _ = check_scores(pipeline, X, y)
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    best_params = study.best_trial.params.copy()
    best_params.update({
        "random_state": random_state,
        "n_jobs": -1
    })
    best_model = XGBRegressor(**best_params)

    best_pipeline = Pipeline([
        ("features", feat_eng_transformer),
        ("model", best_model),
    ])

    return study, best_pipeline


#%%time

#df_train, df_external_train, df_test, df_external_test = load_data()
#X_full = pd.concat([df_train, df_external_train])
#y_full = X_full.pop("BeatsPerMinute")
#X_full = X_full.drop(columns=["id"])

#study, best_pipe = optimize_xgb_regression(X_full, y_full, feat_eng_transformer, n_trials=80)
#print("Best RMSE:", study.best_value)
#print("Best params:", study.best_params)


best_params = {
    'n_estimators': 613, 
    'learning_rate': 0.11463146317047866, 
    'max_depth': 11, 
    'min_child_weight': 0.2172582123588243, 
    'subsample': 0.7256489178989569, 
    'colsample_bytree': 0.5417562490084055, 
    'gamma': 2.8454036986443185, 
    'reg_alpha': 2.7907383513911015e-06, 
    'reg_lambda': 0.01675783822802598, 
    'max_bin': 441, 
    'grow_policy': 'lossguide'
}


df_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
df_submission.head()


%%time
df_train, df_external_train, df_test, df_external_test = load_data()

train_all = pd.concat([df_train, df_external_train], ignore_index=True)

y_full = train_all.pop("BeatsPerMinute")
X_full = train_all.drop(columns=["id"], errors="ignore")

X_test = df_test.drop(columns=["id"], errors="ignore").copy()
ids = df_test["id"].copy()

best_model = XGBRegressor(**best_params)
feat_eng_transformer = FunctionTransformer(feature_engineering, validate=False)

best_pipeline = Pipeline([
    ("features", feat_eng_transformer),
    ("model", best_model),
])

best_pipeline.fit(X_full, y_full)
predictions = best_pipeline.predict(X_test)

# submission
output = pd.DataFrame({
    "id": ids.reset_index(drop=True),
    "BeatsPerMinute": predictions
})
output.to_csv("ps5e9_beats_per_minute_prediction.csv", index=False)
print("Your submission was successfully saved!")

