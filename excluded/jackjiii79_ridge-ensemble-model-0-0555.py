import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

df_train.head()


target_col = 'accident_risk'
exclude_from_features = ['id', target_col]


df_train.info()


df_test.info()


categorical_features = [name for name in df_train.columns if np.issubdtype(df_train.dtypes[name], np.object_)]
df_train[categorical_features].value_counts()


df_train['is_duplicated'], df_test['is_duplicated'] = False, False
duplicate_train_row_ids = np.where(df_train.duplicated(subset=df_train.columns, keep=False))
duplicate_test_row_ids = np.where(df_test.duplicated(subset=df_test.columns, keep=False))

df_train.iloc[duplicate_train_row_ids]['is_duplicated'] = True
df_test.iloc[duplicate_test_row_ids]['is_duplicated'] = True

print(f'duplicated rows in training are: {df_train[df_train["is_duplicated"]]}\nduplicated rows in testing are: {df_test[df_test["is_duplicated"]]}')


df_train.drop('is_duplicated',axis=1, inplace=True)
df_test.drop('is_duplicated', axis=1, inplace=True)


from typing import Sequence
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


def summarize_table_statistics(df: pd.DataFrame, exclude_cols: Sequence[str], outlier_quantile: float = 0.05):
    numeric_features = []
    means = []
    stds = []
    max_list = []
    min_list = []
    trimmed_means = []
    skewnesses = []
    kurtosises = []
    for name in df.columns:
        if np.issubdtype(df.dtypes[name], np.number) and name not in exclude_cols:
            numeric_features.append(name)
            means.append(df[name].mean())
            stds.append(df[name].std())
            max_list.append(df[name].max())
            min_list.append(df[name].min())
            skewnesses.append(skew(df[name]))
            kurtosises.append(kurtosis(df[name]))
            sorted_values = df.loc[~df[name].isna(), name].sort_values()
            boundary_rank = int(len(sorted_values) * outlier_quantile)
            trimmed_means.append(sorted_values[boundary_rank:len(sorted_values)-boundary_rank].mean())

    return pd.DataFrame(
        {
            "mean": means,
            "std": stds,
            "trimmed_mean": trimmed_means,
            "max": max_list,
            "min": min_list,
            "skewness": skewnesses,
            "kurtosis": kurtosises,
        }, index=numeric_features)

for annotation, df in zip(['train', 'test'], [df_train, df_test]):
    summarized_stat_df = summarize_table_statistics(df, exclude_from_features)
    numeric_features = summarized_stat_df.index.tolist()
    print(f"{annotation} dataset summarized statistics:\n{summarized_stat_df}\n")


def plot_mixed_grid(df, columns, is_train=False, col_wrap=3, height=30, bins=30):
    is_num = {c: pd.api.types.is_numeric_dtype(df[c]) for c in columns}
    df_long = df[columns].melt(var_name="column", value_name="value")

    def _per_facet(data, color=None, **kws):
        col = data["column"].iloc[0]
        if is_num[col]:
            sns.histplot(data=data, x="value", kde=True, bins=bins)
            plt.ylabel("Count")
        else:
            order = data["value"].value_counts().index
            sns.countplot(data=data, x="value", order=order)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        plt.xlabel("")

    g = sns.FacetGrid(df_long, col="column", col_wrap=col_wrap,
                      sharex=False, sharey=False, height=height)
    g.map_dataframe(_per_facet)
    g.set_titles("{col_name}", size=55)
    g.fig.suptitle(f"Column Distributions from {'training' if is_train else 'testing'} dataset", y=1.02, fontsize=65)

    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    plt.show()

features = list(set(df_train.columns).difference(set(exclude_from_features)))
plot_mixed_grid(df_train, features, True)
plot_mixed_grid(df_test, features, False)


normalized_numeric_train_data = StandardScaler().fit_transform(df_train[numeric_features])
df_train_normalized = df_train.copy()
df_train_normalized[numeric_features] = normalized_numeric_train_data

g = sns.pairplot(df_train_normalized, vars=numeric_features, corner=True, diag_kind='hist', height=3.2, aspect=1.2)
g.fig.tight_layout()
g.fig.suptitle("Target unaware pair plot")
plt.rcParams["figure.dpi"] = 150
plt.show()


targets = df_train_normalized[target_col]
encoder_map = tuple(zip([np.quantile(targets, 0.25), np.quantile(targets, 0.5), np.quantile(targets, 0.75)], ['Low', 'Meidum', 'High']))

def _encode_category(df):
    for threshold, label in encoder_map:
        if df <= threshold:
            return label
    return "Extreme"

df_train_normalized["risk_level"] = df_train_normalized[target_col].apply(_encode_category)
custom_palette = {}
colors = ['blue', 'green', 'yellow', 'red']
for c_idx, val in enumerate(df_train_normalized['risk_level'].unique()):
    custom_palette[val] = colors[c_idx]
g = sns.pairplot(df_train_normalized, vars=numeric_features, hue='risk_level', palette=custom_palette, corner=True, diag_kind='hist', height=3.2, aspect=1.2)
g.fig.tight_layout()
g.fig.suptitle("Target aware pair plot")
plt.rcParams["figure.dpi"] = 150
plt.show()


from sklearn.preprocessing import FunctionTransformer

original_features = df_test.columns.tolist()
for remove_f in exclude_from_features:
  if remove_f in original_features:
    original_features.remove(remove_f)

print(f"original features: {original_features}")

y_train = df_train[target_col]
df_train = df_train[original_features]

df_train_eng = df_train.copy()
df_test_eng = df_test.copy()[original_features]
for df in [df_train_eng, df_test_eng]:
    curves = df["curvature"]
    curves_threshold = [np.quantile(curves, 0.25), np.quantile(curves, 0.5), np.quantile(curves, 0.75)]
    def generate_road_density_factor(df):
        if df["road_type"] == "urban":
            return df["num_lanes"]
        else:
            return df["num_lanes"] / 2

    def generate_curvature_levels(curve):
        for level, threshold in enumerate(curves_threshold, 1):
            if curve <= threshold:
                return level
        return len(curves_threshold) + 1

    # Sharp curves + high speed -> high accident risk
    df["speed_curvature_index"] = df["speed_limit"] * df["curvature"]

    # Narrower + curvier roads tend to be riskier
    df["lane_curvature_ratio"] = df["curvature"] / df["num_lanes"]

    # Higher speed in poor lighting increases danger
    df["speed_lighting_risk"] = df["speed_limit"] * (df["lighting"].isin(["dim", "night"]).astype(int) + 1)

    # Risk increases when rain/fog + high speed
    df["weather_speed_factor"] = df["speed_limit"] * ((df["weather"] != "clear").astype(int) + 1)

    # Captures school-hour exposure
    df["school_time_zone"] = df["school_season"] & df["time_of_day"].isin(["morning","afternoon"])

    # Normalizes history by lane capacity
    df["accidents_per_lane"] = df["num_reported_accidents"] / df["num_lanes"]

    # Converts into categorical risk grouping
    accident_threshold = 2
    df["is_high_incident_road"] = df["num_reported_accidents"] >= 2

    df["is_poor_lighting"] = df["lighting"].isin(["dim","night"])

    df["is_bad_weather"] = df["weather"] != "clear"

    df["is_peak_hour"] = df["time_of_day"].isin(["morning", "evening"])

    df["is_high_speed_zone"] = df["speed_limit"] > 55

    # Urban lanes does not equivalent to highway lanes
    df["road_density_factor"] = df.apply(generate_road_density_factor, axis=1)

    # Nonlinear
    df["curvature_level"] = df["curvature"].apply(generate_curvature_levels)

    # Absence of signs correlates with risk
    df["is_unmarked_road"] = df["road_signs_present"] == False

    # Public urban areas have pedestrian exposure
    df["is_public_urban"] = df["public_road"] & (df["road_type"] == "urban")

    # Visibility factor
    df["is_night"] = df["time_of_day"] == "evening"

    # Night + speed combined risk
    df["night_speed_interaction"] = df["is_night"].astype(int) * df["speed_limit"]

    # Log transformation
    log_transform = lambda x: np.log1p(x)
    df["log_accidents"] = FunctionTransformer(log_transform, validate=True).fit_transform((df["num_reported_accidents"] + 1).to_numpy().reshape(-1, 1))


df_train_eng.head()


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector


class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.eigenvector_ = None
        self.eigenvalue_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X, y=None):
      assert self.n_components <= X.shape[1]
      self.mean_ = np.mean(X, axis=0)
      normalized_X = X - self.mean_
      U, S, Vt = np.linalg.svd(normalized_X, full_matrices=False)
      U = U[:, :self.n_components]
      S = S[:self.n_components]
      Vt = Vt[:self.n_components, :]
      self.eigenvector_ = Vt.T
      self.eigenvalue_ = S ** 2
      self.explained_variance_ = self.eigenvalue_ / (X.shape[0] - 1)
      total_vars = (normalized_X ** 2).sum() / (X.shape[0] - 1)
      self.explained_variance_ratio_ = self.explained_variance_ / total_vars
      return self

    def transform(self, X):
      assert self.mean_ is not None and self.eigenvector_ is not None
      normalized_X = X - self.mean_
      return normalized_X @ self.eigenvector_

    def inverse_transform(self, X):
      assert self.mean_ is not None and self.eigenvector_ is not None
      return self.mean_ + (X @ self.eigenvector_.T)


df_train_reg = df_train_eng.copy()
y_train_reg = y_train.copy()

orig_numeric_features = list(df_train.select_dtypes(include=np.number).columns)
orig_categorical_features = list(df_train.select_dtypes(include=['object', 'category']).columns)
orig_boolean_features = list(df_train.select_dtypes(include=['bool']).columns)

numeric_features = list(df_train_reg.select_dtypes(include=np.number).columns)
boolean_features = list(df_train_reg.select_dtypes(include=bool).columns)
categorical_features = list(df_train_reg.select_dtypes(include=['object', 'category']).columns)

df_test_final = df_test_eng
for col in categorical_features:
  df_test_final[col] = df_test_final[col].astype('category')

engineer_features = [
    "log_accidents", "night_speed_interaction", "speed_curvature_index", "lane_curvature_ratio", "speed_lighting_risk", "weather_speed_factor",
    "school_time_zone", "accidents_per_lane", "is_high_incident_road", "road_density_factor", "curvature_level", "is_unmarked_road",
    "is_public_urban", "is_night",
]

print(f"Number of numeric_features: {len(numeric_features)}")
print(f"Number of boolean_features: {len(boolean_features)}")
print(f"Number of categorical_features: {len(categorical_features)}")


def cast_int_transformer():
  return FunctionTransformer(lambda x: x.astype(int))

def build_baseline_pipeline(alpha = 1.0, n_components = 10, random_state:int = 1234):
  pipe_def = []
  numeric_pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('pca', PCATransformer(n_components=n_components))
  ])
  categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  boolean_encoder = cast_int_transformer()
  preprocessor = ColumnTransformer(
      transformers=[
          ("numeric", numeric_pipe, make_column_selector(dtype_include=np.number)),
          ("categorical", categorical_encoder, make_column_selector(dtype_include=['object', 'category'])),
          ("boolean", boolean_encoder, make_column_selector(dtype_include=['bool'])),
      ]
  )
  pipe_def.append(('preprocessor', preprocessor))
  pipe_def.append(('estimator', Ridge(alpha=alpha, fit_intercept=False)))
  return Pipeline(pipe_def)


!pip install catboost
!pip install optuna "optuna-integration[xgboost]" "optuna-integration[lightgbm]"
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback
from sklearn.metrics import mean_squared_error


# Find best n_components
random_state = 1234
linear_cv_scores = []
def objective(trial, df_train, y_train, num_features, seed):
  pipeline = build_baseline_pipeline(alpha=trial.suggest_float("alpha", 0.05, 10), n_components=trial.suggest_int("n_components", 3, num_features), random_state=seed)
  kf = KFold(n_splits=10, shuffle=True, random_state=1234)
  scores = cross_val_score(pipeline, df_train, y_train, cv=kf, n_jobs=1, scoring='neg_root_mean_squared_error')
  mean_rmse = float(-np.mean(scores))
  linear_cv_scores.append({
      "trail": trial.number,
      "params": trial.params,
      "mean_rmse": mean_rmse,
  })
  return mean_rmse


linear_eng_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
linear_eng_study.optimize(lambda tr: objective(tr, df_train_reg, y_train, len(numeric_features), seed=random_state), n_trials=60, n_jobs=1, show_progress_bar=True)
print(f"With Engineered features:")
print(f"Best Trail: {linear_eng_study.best_trial.number}")
print(f"Best Score: {linear_eng_study.best_trial.value}")
print(f"Best Params: {linear_eng_study.best_trial.params}")
print(f"CV SCORES:")
for row in linear_cv_scores:
  print(f"Trail: {row['trail']}, Params: {row['params']}, Mean RMSE: {row['mean_rmse']}")


random_state = 1234
def kfold(y, n_splits=10, seed=1234):
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
  for train_index, val_index in kf.split(y):
    yield train_index, val_index


catboost_cv_scores = []
df_train_catboost = df_train_reg.copy()
for col in categorical_features:
  df_train_catboost[col] = df_train_catboost[col].astype('category')
y_train_catboost = y_train.copy()
cat_idx_for_cb = [idx for idx, col in enumerate(df_train_catboost.columns) if col in categorical_features]

def objective(trial, X, y, cat_cols, bool_cols, num_cols, cat_idx_cb, n_splits=5, seed=1234):
  early_stopping_rounds = 200
  params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",

    "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),

    "iterations": trial.suggest_int("iterations", 1500, 3500),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "depth": trial.suggest_int("depth", 4, 12),

    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),

    "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),

    "border_count": trial.suggest_int("border_count", 32, 255),

    "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),

    "random_seed": seed,
    "allow_writing_files": False,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0",
  }
  if params["bootstrap_type"] == "Bayesian":
    params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
  if params["bootstrap_type"] == "Bernoulli":
    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

  rmses = []
  for fold, (tr_idx, va_idx) in enumerate(kfold(y, n_splits, seed)):
    X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx_cb)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx_cb)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds, use_best_model=True)
    pred = model.predict(val_pool)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    if trial.should_prune():
      raise optuna.TrialPruned()
    rmses.append(rmse)

  catboost_cv_scores.append({
    "trail": trial.number,
    "params": trial.params,
    "mean_rmse": float(np.mean(rmses)),
  })
  return float(np.mean(rmses))

## Skip tuning as time consuming
#catboost_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
#catboost_study.optimize(lambda tr: objective(tr, df_train_catboost, y_train_catboost, categorical_features, boolean_features, numeric_features, cat_idx_for_cb, n_splits=5, seed=random_state), n_trials=60, n_jobs=1, show_progress_bar=True)
catboost_best_params = {
    'boosting_type': 'Plain',
    'iterations': 3359,
    'learning_rate': 0.027488273782026087,
    'depth': 7,
    'l2_leaf_reg': 0.6737709240195966,
    'bootstrap_type': 'Bernoulli',
    'border_count': 105,
    'random_strength': 0.00955446721503038,
    'subsample': 0.8372219456283703,
}
#print(f"Best Trail: {catboost_study.best_trial.number}")
#print(f"Best Score: {catboost_study.best_trial.value}")
#print(f"Best Params: {catboost_study.best_trial.params}")
#print(f"CV SCORES:")
#for row in catboost_cv_scores:
#  print(f"Trail: {row['trail']}, Params: {row['params']}, Mean RMSE: {row['mean_rmse']}")


import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The reported value is ignored because this `step` .* is already reported\.",
    category=UserWarning,
    module=r"optuna\.trial\._trial",
)

lightgbm_cv_scores = []
df_train_lightgbm = df_train_catboost.copy()
y_train_lightgbm = y_train_catboost.copy()
def objective(trial, X, y, cat_cols, bool_cols, num_cols, n_splits=10, seed=1234):
  early_stopping_rounds = 200
  params = {
        "objective": "rmse",
        "metric": "rmse",

        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 16),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),

        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),

        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),

        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),

        "max_bin": trial.suggest_int("max_bin", 63, 255),

        "force_col_wise": True,
        "device_type": "gpu",
        "verbosity": -1,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
  }

  rmses = []
  for fold, (tr_idx, va_idx) in enumerate(kfold(y, n_splits, seed)):
      X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
      y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
      lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns), categorical_feature=cat_cols, free_raw_data=True)
      lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=list(X_val.columns), categorical_feature=cat_cols, free_raw_data=True, reference=lgb_train)
      pruning_cb = LightGBMPruningCallback(trial, "rmse")
      model = lgb.train(params, lgb_train, valid_sets=[lgb_val],
                      num_boost_round=trial.suggest_int("n_estimators", 1500, 4500),
                      callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False), pruning_cb])
      pred = model.predict(X_val)
      rmse = root_mean_squared_error(y_val, pred)
      rmses.append(rmse)
      trial.report(rmse, step=fold)
      if trial.should_prune():
        raise optuna.TrialPruned()
  mean_rmse = float(np.mean(rmses))
  lightgbm_cv_scores.append({
      "trail": trial.number,
      "params": trial.params,
      "mean_rmse": mean_rmse,
      "fold_rmse": rmses,
  })
  return mean_rmse

lightgbm_best_params = {
    'num_leaves': 237,
    'max_depth': 16,
    'feature_fraction': 0.7961716897803867,
    'bagging_fraction': 0.9422102010301452,
    'bagging_freq': 4,
    'learning_rate': 0.2931821442725745,
    'lambda_l1': 0.0018906563051563048,
    'lambda_l2': 0.0018000250626364516,
    'min_data_in_leaf': 23,
    'min_gain_to_split': 0.0025873554436450716,
    'min_sum_hessian_in_leaf': 0.01440451338043205,
    'max_bin': 255,
    'n_estimators': 1684,
}
## Skip as time consuming
#lightgbm_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
#lightgbm_study.optimize(lambda tr: objective(tr, df_train_lightgbm, y_train_lightgbm, categorical_features, boolean_features, numeric_features, n_splits=10, seed=random_state), n_trials=60, n_jobs=1, show_progress_bar=True)
#print(f"Best Trail: {lightgbm_study.best_trial.number}")
#print(f"Best Score: {lightgbm_study.best_trial.value}")
#print(f"Best Params: {lightgbm_study.best_trial.params}")
#print(f"CV SCORES:")
#for row in lightgbm_cv_scores:
#  print(f"Trail: {row['trail']}, Params: {row['params']}, Mean RMSE: {row['mean_rmse']}")


xgboost_cv_scores = []
df_train_xgboost = df_train_lightgbm.copy()
y_train_xgboost = y_train_lightgbm.copy()
def objective(trial, X, y, cat_cols, bool_cols, num_cols, n_splits=10, seed=1234):
  early_stopping_rounds = 200
  params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",

        "tree_method": "gpu_hist",
        "predictor":   "gpu_predictor",
        "max_bin": trial.suggest_int("max_bin", 63, 512),
        "device": "cuda",

        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "max_leaves": trial.suggest_int("max_leaves", 0, 512),

        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
        "alpha":  trial.suggest_float("alpha",  1e-3, 10.0,  log=True),

        "eta": trial.suggest_float("eta", 1e-4, 0.2, log=True),
        "verbosity": 0
  }

  rmses = []
  for fold, (tr_idx, va_idx) in enumerate(kfold(y, n_splits, seed)):
    X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    pruning_cb = XGBoostPruningCallback(trial, "valid-rmse")
    model = xgb.train(params, dtrain, evals=[(dval, "valid")],
                      num_boost_round=trial.suggest_int("n_estimators", 1500, 3500),
                      callbacks=[pruning_cb], verbose_eval=False, early_stopping_rounds=early_stopping_rounds)
    pred = model.predict(dval)
    rmse = root_mean_squared_error(y_val, pred)
    rmses.append(rmse)
    trial.report(rmse, step=fold)
    if trial.should_prune():
      raise optuna.TrialPruned()
  mean_rmse = float(np.mean(rmses))
  xgboost_cv_scores.append({
      "trail": trial.number,
      "params": trial.params,
      "mean_rmse": mean_rmse,
      "fold_rmse": rmses,
  })
  return mean_rmse

xgboost_best_params = {
    'max_bin': 478,
    'max_depth': 11,
    'min_child_weight': 0.1222406875368088,
    'gamma': 0.009114080279666419,
    'grow_policy': 'lossguide',
    'max_leaves': 406,
    'subsample': 0.8073664094355828,
    'colsample_bytree': 0.9912833374156531,
    'colsample_bylevel': 0.6794271002629645,
    'lambda': 4.835901587555031,
    'alpha': 0.0035926447269045945,
    'eta': 0.09917239529678558,
    'n_estimators': 2473,
}
## Skip as time consuming
#xgboost_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
#xgboost_study.optimize(lambda tr: objective(tr, df_train_xgboost, y_train_xgboost, categorical_features, boolean_features, numeric_features, n_splits=10, seed=random_state), n_trials=60, n_jobs=1, show_progress_bar=True)
#print(f"Best Trail: {xgboost_study.best_trial.number}")
#print(f"Best Score: {xgboost_study.best_trial.value}")
#print(f"Best Params: {xgboost_study.best_trial.params}")
#print(f"CV SCORES:")
#for row in xgboost_cv_scores:
#  print(f"Trail: {row['trail']}, Params: {row['params']}, Mean RMSE: {row['mean_rmse']}")


from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.base import RegressorMixin
from pandas.api.types import CategoricalDtype, is_categorical_dtype

def lock_categorical_dtypes(df, cat_cols):
    df = df.copy()
    cat_types = {}
    for c in cat_cols:
        cats = sorted(map(str, pd.Series(df[c], dtype="object").dropna().unique()))
        cat_types[c] = CategoricalDtype(categories=cats, ordered=False)
        df[c] = df[c].astype(cat_types[c])
    return df, cat_types

class LightGBMRegressorWrapper(RegressorMixin, BaseEstimator):
    _estimator_type = "regressor"
    def __init__(self, features=None, cat_features=None, **lgbm_params):
        self.cat_features = cat_features
        self.features = features
        self.lgbm_params = lgbm_params
        self.model_ = None
        self._cat_types_ = {}

    def _coerce_predict_frame(self, X):
        Xc = X.copy()
        Xc = Xc[self.features]
        for c, dt in self._cat_types_.items():
            Xc[c] = Xc[c].astype(dt)
        return Xc

    def fit(self, X, y, **fit_params):
        feature_name = self.features if self.features is not None else list(X.columns)
        categorical_feature = self.cat_features if self.cat_features is not None else [c for c in feature_name if is_categorical_dtype(X[c])]
        self._cat_types_.clear()
        for c in categorical_feature:
            self._cat_types_[c] = X[c].dtype

        self.model_ = lgb.LGBMRegressor(**self.lgbm_params)
        self.model_.fit(
            X, y,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
        )
        self.features = feature_name
        self.cat_features = categorical_feature
        return self

    def predict(self, X):
        Xc = self._coerce_predict_frame(X)
        return self.model_.predict(Xc)

    def get_params(self, deep: bool = True):
        return {"cat_features": self.cat_features, "features": self.features, **self.lgbm_params}

    def set_params(self, **params) -> "LightGBMRegressorWrapper":
        if "cat_features" in params:
            self.cat_features = params.pop("cat_features")
        if "features" in params:
            self.features = params.pop("features")
        if params:
            self.lgbm_params.update(params)
        return self


class CatBoostRegressorWrapper(RegressorMixin, BaseEstimator):
    _estimator_type = "regressor"
    def __init__(self, cat_features_idx=None, **cb_params):
        self.cat_features_idx = cat_features_idx
        self.cb_params = cb_params
        self.model_ = None

    def fit(self, X, y, **fit_params):
        cat_features_idx = self.cat_features_idx if self.cat_features_idx is not None else [idx for idx, c in enumerate(X.columns) if pd.api.types.is_categorical_dtype(X[c])]
        pool = Pool(X, y, cat_features=self.cat_features_idx)
        self.model_ = CatBoostRegressor(**self.cb_params)
        self.model_.fit(pool, verbose=False)
        self.cat_features_idx = cat_features_idx
        return self

    def predict(self, X):
        pool = Pool(X, cat_features=self.cat_features_idx)
        return self.model_.predict(pool)

    def get_params(self, deep: bool = True):
        return {"cat_features_idx": self.cat_features_idx, **self.cb_params}

    def set_params(self, **params) -> "CatBoostRegressorWrapper":
        if "cat_features_idx" in params:
            self.cat_features_idx = params.pop("cat_features_idx")
        if params:
            self.cb_params.update(params)
        return self


from sklearn.base import is_regressor, clone

stacking_cv_scores = []


def make_base_models(catboost_params, lightgbm_params, xgboost_params, features, cat_features, cat_idx_for_cb, seed = 1234):
  linear_model = build_baseline_pipeline(alpha=0.7530154387691597, n_components=13, random_state=seed)
  
  catboost_model = CatBoostRegressorWrapper(cat_features_idx=cat_idx_for_cb, **catboost_params)
  lightgbm_model = LightGBMRegressorWrapper(features=features, cat_features=cat_features, **lightgbm_params)

  assert is_regressor(catboost_model)
  assert is_regressor(lightgbm_model)
  xgboost_model = xgb.XGBRegressor(enable_categorical=True, **xgboost_params)
  return [
      ("linear", linear_model),
      ("catboost", catboost_model),
      ("lightgbm", lightgbm_model),
      ("xgboost", xgboost_model)
  ]

def make_stacking_model(base_models, alpha=0.1, n_splits=3, seed=1234):
  meta = Ridge(alpha=alpha)
  return StackingRegressor(estimators=base_models, final_estimator=meta, passthrough=False, n_jobs=None)

def make_oof_predictions(X, y, base_models, n_splits=10, seed=1234):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), len(base_models)))
    for fold, (tr, va) in enumerate(kf.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y.iloc[tr]
        for j, (_, model) in enumerate(base_models):
            m = clone(model)  # sklearn clone
            m.fit(X_tr, y_tr)
            oof[va, j] = m.predict(X_va)
    return oof

df_train_final = df_train_xgboost.copy()
y_train_final = y_train_xgboost.copy()

#catboost_best_params = catboost_study.best_params
catboost_best_params["loss_function"] = "RMSE"
catboost_best_params["eval_metric"] = "RMSE"
catboost_best_params["random_seed"] = random_state
catboost_best_params["task_type"] = "GPU"
catboost_best_params["devices"] = "0"
catboost_best_params["verbose"] = False
catboost_best_params["allow_writing_files"] = False
print(f"params for catboost model: {catboost_best_params}")

#lightgbm_best_params = lightgbm_study.best_params
lightgbm_best_params["objective"] = "rmse"
lightgbm_best_params["metric"] = "rmse"
lightgbm_best_params["device_type"] = "gpu"
lightgbm_best_params["gpu_platform_id"] = 0
lightgbm_best_params["gpu_device_id"] = 0
lightgbm_best_params["force_col_wise"] = True
lightgbm_best_params["random_state"] = random_state
lightgbm_best_params["verbosity"] = -1
print(f"params for lightgbm model: {lightgbm_best_params}")

#xgboost_best_params = xgboost_study.best_params
xgboost_best_params["objective"] = "reg:squarederror"
xgboost_best_params["eval_metric"] = "rmse"
xgboost_best_params["tree_method"] = "hist"
xgboost_best_params["predictor"] = "predicator"
xgboost_best_params["device"] = "cpu"
xgboost_best_params["random_state"] = random_state
xgboost_best_params["verbosity"] = 0
print(f"params for xgboost model: {xgboost_best_params}")

#base_models = make_base_models(catboost_best_params, lightgbm_best_params, xgboost_best_params, list(df_train_final.columns), categorical_features, cat_idx_for_cb)
#print("Generating OOF predictions...")
#oof_preds = make_oof_predictions(df_train_final, y_train_final, base_models, n_splits=10)

def objective(trial, seed):
    alpha = trial.suggest_float("alpha", 0.1, 100.0)
    meta = Ridge(alpha=alpha)
    scores = cross_val_score(
        meta,
        oof_preds,
        y_train_final,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return -scores.mean()

meta_best_params = {'alpha': 0.6645023275808258}
## skip as time consuming
#print("Generated OOF predictions done.")
#stacking_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
#print("Training final model...")
#stacking_study.optimize(lambda tr: objective(tr, seed=random_state), n_trials=100, n_jobs=1, show_progress_bar=True)
#print(f"Best Trail: {stacking_study.best_trial.number}")
#print(f"Best Score: {stacking_study.best_trial.value}")
#print(f"Best Params: {stacking_study.best_trial.params}")
#print(f"Stacking score:\n {stacking_cv_scores}")
# meta_best_params = stacking_study.best_trial.params


print(f"Training final model...")
base_models = make_base_models(catboost_best_params, lightgbm_best_params, xgboost_best_params, list(df_train_final.columns), categorical_features, cat_idx_for_cb)
stack = make_stacking_model(base_models, alpha=meta_best_params["alpha"],seed=random_state)
stack.fit(df_train_final, y_train_final)


print(f"Predicting final test set...")
pred = stack.predict(df_test_final)

id_col = df_test["id"]
final_results = pd.DataFrame({"id":id_col, "accident_risk": pred})
final_results.to_csv("/kaggle/working/submission.csv")
final_results.head()

