import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base            import BaseEstimator, RegressorMixin



train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv'), pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
train.head()


# Assuming train and test DataFrames are already defined
# Create DataFrames
pro_train = pd.DataFrame()
pro_test = pd.DataFrame()

# Fill DataFrames with data
pro_train['Genre'] = train['Genre']
pro_test['Genre'] = test['Genre']
pro_train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage']
pro_train['Host_Popularity_percentage'] = train['Host_Popularity_percentage']
y = train['Listening_Time_minutes']
pro_test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage']
pro_test['Host_Popularity_percentage'] = test['Host_Popularity_percentage']
pro_train['polynomial'] = (pro_train['Guest_Popularity_percentage']**2) + (pro_train['Guest_Popularity_percentage'] * pro_train['Host_Popularity_percentage']) + (pro_train['Host_Popularity_percentage']**2)
pro_test['polynomial'] =(pro_test['Guest_Popularity_percentage']**2) + (pro_test['Guest_Popularity_percentage'] * pro_test['Host_Popularity_percentage']) + (pro_test['Host_Popularity_percentage']**2)


day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_map = {'Night':0, 'Afternoon':1, 'Evening':2, 'Morning':3}
sentiment_map = {'Positive':0, 'Negative':2 ,'Neutral':1,}

pro_train['Episode_Length_minutes']=train['Episode_Length_minutes']
pro_test['Episode_Length_minutes']=test['Episode_Length_minutes']
pro_train['Number_of_Ads']=train['Number_of_Ads']
pro_test['Number_of_Ads']=test['Number_of_Ads']

pro_test['Pub_time'] = test['Publication_Time'].map(time_map)
pro_train['Pub_time'] = train['Publication_Time'].map(time_map)
pro_train['day_num'] = train['Publication_Day'].map(day_map)
pro_test['day_num'] = test['Publication_Day'].map(day_map)
pro_test['Sentiment'] = test['Episode_Sentiment'].map(sentiment_map) # 0 is positivve 1 neutral and 2 negative
pro_train['Sentiment'] = train['Episode_Sentiment'].map(sentiment_map)
unique_podcasts = train['Podcast_Name'].unique()
podcast_dict = dict(enumerate(unique_podcasts))
pro_test['Podcast_Name'] = test['Podcast_Name'].map(podcast_dict) # 0 is positivve 1 neutral and 2 negative
pro_train['Podcast_Name'] = train['Podcast_Name'].map(podcast_dict)

pro_train['day_sin'] = np.sin(2 * np.pi * pro_train['day_num'] / 7)
pro_train['day_cos'] = np.cos(2 * np.pi * pro_train['day_num'] / 7)
pro_train['Pub_sin'] = np.sin(2 * np.pi * pro_train['Pub_time'] / 4)
pro_train['Pub_cos'] = np.cos(2 * np.pi * pro_train['Pub_time'] / 4)

pro_test['Pub_sin'] = np.sin(2 * np.pi * pro_test['Pub_time'] / 4)
pro_test['Pub_cos'] = np.cos(2 * np.pi * pro_test['Pub_time'] / 4)
pro_test['day_sin'] = np.sin(2 * np.pi * pro_test['day_num'] / 7)
pro_test['day_cos'] = np.cos(2 * np.pi * pro_test['day_num'] / 7)

pro_train.columns


import warnings
from tqdm.auto import tqdm

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def num_preprocessor():
    return MinMaxScaler()

def tune_lightgbm_with_optuna(X, y, num_features, n_trials=20, n_splits=3, random_state=42):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 64),
            'random_state': random_state,
        }
        preproc = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )
        pipe = Pipeline([
            ("preprocessor", preproc),
            ("regressor", lgb.LGBMRegressor(**params)),
        ])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_score = np.sqrt(-cv_scores.mean())
        return rmse_score

    study = optuna.create_study(direction="minimize")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # Optuna progress bar
    return study.best_params

def train_per_category_lightgbm_models_optuna(
    X, y, cat_feature, num_features,
    test_size=0.20, random_state=42, n_splits=5, n_trials=20):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=X[cat_feature]
    )

    models_dict = {}
    summary_rows = []

    cats = sorted(X_train[cat_feature].unique())

    # tqdm category progress bar
    for cat in tqdm(cats, desc="Optimizing categories", leave=True):
        idx_tr = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature] == cat
        
        X_tr_cat = X_train.loc[idx_tr, :]
        y_tr_cat = y_train.loc[idx_tr]
        X_val_cat = X_val.loc[idx_val, :]
        y_val_cat = y_val.loc[idx_val]

        # ----> Optuna search for best params (with warnings suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_params = tune_lightgbm_with_optuna(
                X_tr_cat, y_tr_cat, num_features,
                n_trials=n_trials, n_splits=3, random_state=random_state
            )

        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )
        pipe_cat = Pipeline([("preprocessor", preproc_cat),
                             ("regressor", lgb.LGBMRegressor(**best_params))])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_cat.fit(X_tr_cat, y_tr_cat)
            y_pred_val = pipe_cat.predict(X_val_cat)
        
        cat_rmse = rmse(y_val_cat, y_pred_val)

        models_dict[cat] = {"model": pipe_cat, "rmse": cat_rmse, "best_params": best_params}
        summary_rows.append(dict(Genre=cat, RMSE=cat_rmse))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)



class MultiGenreLightGBMModel(BaseEstimator, RegressorMixin):
    """
    One LightGBM Regressor per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42, n_trials=20):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.n_trials            = n_trials
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_lightgbm_models_optuna(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state,
                                           n_trials    = self.n_trials)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        preds = pd.Series(index=X.index, dtype=float)
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred
        return preds.values

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}


cat_col  = "Genre"
num_cols = ['Guest_Popularity_percentage', 'Host_Popularity_percentage',
       'polynomial', 'Episode_Length_minutes', 'Number_of_Ads', 'Pub_time',
       'day_num', 'Sentiment', 'Podcast_Name', 'day_sin', 'day_cos', 'Pub_sin',
       'Pub_cos']
model_lightgbm = MultiGenreLightGBMModel(
    categorical_feature=cat_col, 
    numerical_features=num_cols, 
    test_size=0.2, 
    random_state=42, 
    n_trials=100  # or higher for better results, e.g. 30 or 50
)

model_lightgbm.fit(pro_train, y)


submission.sample(90)


model_lightgbm.per_category_rmse()


import xgboost as xgb
def tune_xgboost_with_optuna(X, y, num_features, n_trials=20, n_splits=3, random_state=42):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
            'verbosity': 0,
            'objective': 'reg:squarederror',
        }
        preproc = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )
        pipe = Pipeline([
            ("preprocessor", preproc),
            ("regressor", xgb.XGBRegressor(**params)),
        ])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_score = np.sqrt(-cv_scores.mean())
        return rmse_score

    study = optuna.create_study(direction="minimize")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def train_per_category_xgboost_models_optuna(
    X, y, cat_feature, num_features,
    test_size=0.20, random_state=42, n_splits=3, n_trials=20):
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=X[cat_feature]
    )

    models_dict = {}
    summary_rows = []

    cats = sorted(X_train[cat_feature].unique())

    for cat in tqdm(cats, desc="Optimizing categories (XGBoost)", leave=True):
        idx_tr = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature] == cat
        
        X_tr_cat = X_train.loc[idx_tr, :]
        y_tr_cat = y_train.loc[idx_tr]
        X_val_cat = X_val.loc[idx_val, :]
        y_val_cat = y_val.loc[idx_val]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_params = tune_xgboost_with_optuna(
                X_tr_cat, y_tr_cat, num_features,
                n_trials=n_trials, n_splits=n_splits, random_state=random_state
            )

        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )
        pipe_cat = Pipeline([
            ("preprocessor", preproc_cat),
            ("regressor", xgb.XGBRegressor(**best_params))
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_cat.fit(X_tr_cat, y_tr_cat)
            y_pred_val = pipe_cat.predict(X_val_cat)

        cat_rmse = rmse(y_val_cat, y_pred_val)

        models_dict[cat] = {"model": pipe_cat, "rmse": cat_rmse, "best_params": best_params}
        summary_rows.append(dict(Genre=cat, RMSE=cat_rmse))
    
    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)


class MultiGenreXGBoostModel(BaseEstimator, RegressorMixin):
    """
    One XGBoost Regressor per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42, n_trials=20):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.n_trials            = n_trials
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_xgboost_models_optuna(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state,
                                           n_trials    = self.n_trials)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        preds = pd.Series(index=X.index, dtype=float)
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred
        return preds.values

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}


model_xgbm = MultiGenreXGBoostModel(
    categorical_feature=cat_col, 
    numerical_features=num_cols, 
    test_size=0.2, 
    random_state=42, 
    n_trials=100  # or higher for better results, e.g. 30 or 50
)

model_xgbm.fit(pro_train, y)


model_xgbm.per_category_rmse()


def compute_weights(rmse_1, rmse_2):
    w1 = 1 / rmse_1
    w2 = 1 / rmse_2
    total = w1 + w2
    return w1 / total, w2 / total


y_test_pred_lightgbm = model_lightgbm.predict(pro_test)
y_test_model_xgbm = model_xgbm.predict(pro_test)


lgbm_rmse = model_lightgbm.per_category_rmse() 
xgb_rmse = model_xgbm.per_category_rmse()

# Compute weights dict for each category
weights = {}
for cat in lgbm_rmse.keys():
    w_lgbm, w_xgb = compute_weights(lgbm_rmse[cat], xgb_rmse[cat])
    weights[cat] = (w_lgbm, w_xgb)

# Prepare preds series
blended_preds = pd.Series(index=pro_test.index, dtype=float)

# For each category, blend predictions using the computed weights
for cat, idx in pro_test.groupby(cat_col).groups.items():
    w_lgbm, w_xgb = weights.get(cat, (0.5, 0.5))  # default to equal if missing
    preds_lgbm_cat = y_test_pred_lightgbm[idx]
    preds_xgb_cat = y_test_model_xgbm[idx]
    
    blended_preds.loc[idx] = w_lgbm * preds_lgbm_cat + w_xgb * preds_xgb_cat

# Convert to numpy array if needed
y_pred_blended = blended_preds.values

submission['Listening_Time_minutes'] = y_pred_blended
plt.boxplot(submission['Listening_Time_minutes'])
submission.to_csv('multi_model_xgcombine.csv', index=False)




