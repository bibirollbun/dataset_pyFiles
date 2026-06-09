import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"


warnings.simplefilter(action='ignore', category=FutureWarning)


PATH = '/kaggle/input/playground-series-s5e8/'


test = pd.read_csv(PATH + 'test.csv')
test.head()


train = pd.read_csv(PATH + 'train.csv')
train.head()


sample_sub = pd.read_csv(PATH + 'sample_submission.csv')
sample_sub.head()


original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
original['y'] = [1 if row == 'yes' else 0 for row in original['y']]
original.head()


# joining train and original dataset
df = pd.concat([train.drop('id', axis=1), original]).reset_index(drop=True)
# df = train.copy()
df.head()


df.describe()


sns.countplot(df, x='y') # imbalanced class


df.isna().any()


for col in df[df.select_dtypes(include='number').columns]:
    print(f"{col}:{np.isinf(df[col]).any()}")


# understanding cols
for col in df.columns:
    if df[col].dtype == 'O':
        print(f"Col: {col} | Values: {df[col].unique()}")
    else:
        print(f"Col: {col} is numeric.")


cat_cols = df.select_dtypes(exclude='number').columns
df[cat_cols]


for col in cat_cols:
    plt.figure(figsize=(12,8))
    sns.countplot(df, x=col)
    plt.show()


numerical_cols = df.select_dtypes(include='number').columns
df[numerical_cols]


for col in numerical_cols:
    plt.figure(figsize=(12,8))
    sns.histplot(data=df[col])
    plt.title(f'Hist for {col}')
    plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(numeric_only=True),
    vmax=1,
    vmin=-1,
    annot=True,
    fmt='.1f'
)


class FeatureEngineering:
    def __init__(self):
        self.binary_cols = ['default', 'housing', 'loan']

    def transform_binary_cols(self, df):
        for col in self.binary_cols:
            df[col] = [1 if row == 'yes' else 0 for row in df[col]]

    def frequency_encode(self, df, cols):
        for col in cols:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)

    def create_cat_cols(self, df):
        df['balance_is_pos'] = [1 if row >= 0 else 0 for row in df['balance']]
        df['had_previous'] = [1 if row > -1 else 0 for row in df['pdays']]
        df['campaign_intensity'] = pd.cut(df['campaign'], bins=[0, 1, 3, 10, np.inf], labels=['low', 'medium', 'high', 'extreme'])
        df['previous_succeed'] = [1 if row == 'success' else 0 for row in df['poutcome']]

    def create_combinations(self, df):
        df['duration_per_campaign'] = df['duration'] / df['campaign']

        unknowns = ['poutcome', 'contact', 'education', 'job']
        for i in unknowns:
            df[f'{i}_known'] = [1 if row != 'unknown' else 0 for row in df[i]]

        for i in self.binary_cols:
            if i != 'default':
                df[f'{i}_comb_default'] = df[i] & df['default']
        df['housing_comb_loan'] = df['housing'] & df['loan']

    def log_cols(self, df):
        df['balance']  = np.log1p(df['balance'] - df['balance'].min() + 1)
        df['duration'] = np.log1p(df['duration'])
        
    def day_sin_cos(self, df):
        df['day_sin'] = np.sin(df['day'])
        df['day_cos'] = np.cos(df['day'])
        df.drop('day', axis=1, inplace=True)

    def run(self, df):
        _df = df.copy()
        cat_cols = _df.select_dtypes(exclude='number').columns
        self.transform_binary_cols(_df)
        self.frequency_encode(_df, cat_cols)
        self.create_cat_cols(_df)
        self.create_combinations(_df)
        self.log_cols(_df)
        _df.drop('pdays', axis=1, inplace=True)
        # _df['pdays'] = [0 if row == -1 else row for row in _df['pdays']]
        _df = pd.get_dummies(_df, columns=_df.select_dtypes(exclude='number').columns, drop_first=True)
        self.day_sin_cos(_df)
        return _df


feat_eng = FeatureEngineering()
_y = df['y']
# to_feat_df = df.drop(['y', 'id'], axis=1)
to_feat_df = df.copy()
featured_df = feat_eng.run(to_feat_df)
featured_df['y'] = _y
featured_df.head()


featured_df_train = featured_df[:len(train)]
# featured_df_original = featured_df[len(train):]


target = 'y'
X = featured_df.drop(target, axis=1)
y = featured_df[target]

# X_original = featured_df_original.drop(target, axis=1)
# y_original = featured_df_original[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
#     X_original,
#     y_original,
#     test_size=0.2,
#     random_state=42,
# )


X_train.head()


# X_train_original.head()


# def xgb_objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
#         'max_leaves' : trial.suggest_int('max_leaves', 50, 800),
#         'max_depth': trial.suggest_int('max_depth', 0, 7),
#         # 'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 3.0),
#         'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.6),
#         'subsample': trial.suggest_float('subsample', 0.4, 0.8),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
#         # 'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.5, 0.9),
#         # 'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.75, 0.95),
#         # 'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('alpha', 1.0, 5.0),
#         # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
#     }
    
#     rocauc = []
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     for train_index, test_index in cv.split(X_train, y_train):
#         X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
#         y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#         xgb = XGBClassifier(
#             **params,
#             eval_metric='auc',
#             sampling_method='gradient_based' if device == 'cuda' else 'uniform',
#             grow_policy='lossguide', 
#             tree_method='hist',
#             device=device,
#             objective='binary:logistic',
#             n_jobs=-1,
#             random_state=42,
#         )

#         xgb.fit(
#             X_train_fold,
#             y_train_fold,
#             eval_set=[(X_test_fold, y_test_fold)],
#             verbose=400,
#         )
#         y_pred_fold = xgb.predict_proba(X_test_fold)[:, 1]

#         _rocauc = roc_auc_score(y_test_fold, y_pred_fold)
#         rocauc.append(_rocauc)
#     return np.mean(rocauc)

# xgb_study = optuna.create_study(direction='maximize', study_name='xgb_study')
# xgb_study.optimize(xgb_objective, n_trials=5)
# xgb_best_params = xgb_study.best_params
# print(f"Best parameters: {xgb_best_params}")


# optuna.visualization.plot_optimization_history(xgb_study).show()


# optuna.visualization.plot_param_importances(xgb_study).show()


# optuna.visualization.plot_slice(xgb_study).show()


# xgb_best_params = {
#     'n_estimators': 617,
#     'max_leaves': 362,
#     'max_depth': 6,
#     'learning_rate': 0.15045447156235414,
#     'subsample': 0.6540672112816384,
#     'colsample_bytree': 0.7858696972402596,
#     'alpha': 1.8866566763680603
# }


# xgb_model = XGBClassifier(
#     **xgb_best_params,
#     eval_metric='auc',
#     sampling_method='gradient_based' if device == 'cuda' else 'uniform',
#     grow_policy='lossguide', 
#     tree_method='hist',
#     device=device,
#     objective='binary:logistic',
#     n_jobs=-1,
#     random_state=42,
# )
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict_proba(X_test)[:, 1]

# print(f'AUC ROC score: {roc_auc_score(y_test, y_pred)}')


# def lgbm_objective(trial):
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 5, 80),
#         'num_leaves': trial.suggest_int('num_leaves', 50, 500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4),
#         'n_estimators': trial.suggest_int('n_estimators', 200, 30000),
#         'min_child_samples': trial.suggest_int('min_child_samples', 0, 60),
#         'subsample': trial.suggest_float('subsample', 0.3, 0.8),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.2, 3.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 7.0),
#         'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 0.6),
#     }

#     rocauc = []
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     for train_index, test_index in cv.split(X_train, y_train):
#         X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
#         y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#         lgbm_model = LGBMClassifier(
#             **params,
#             verbose=-1,
#             objective='binary',
#             device='gpu' if device == 'cuda' else 'cpu',
#             random_state=42,
#         )

#         lgbm_model.fit(
#             X_train_fold, 
#             y_train_fold, 
#             eval_metric='auc',
#             eval_set=[(X_test_fold, y_test_fold)],
#         )
#         y_pred_fold = lgbm_model.predict_proba(X_test_fold)[:, 1]

#         _rocauc = roc_auc_score(y_test_fold, y_pred_fold)
#         rocauc.append(_rocauc)
#     return np.mean(rocauc)

# lgbm_study = optuna.create_study(direction='maximize', study_name='lgbm_study')
# lgbm_study.optimize(lgbm_objective, n_trials=5)
# lgbm_best_params = lgbm_study.best_params
# print(f"Best parameters: {lgbm_best_params}")


# optuna.visualization.plot_optimization_history(lgbm_study).show()


# optuna.visualization.plot_param_importances(lgbm_study).show()


# optuna.visualization.plot_slice(lgbm_study).show()


lgbm_best_params = {
     'max_depth': 10,
     'num_leaves': 100,
     'learning_rate': 0.05509815743323612,
     'n_estimators': 25000,
     'subsample': 0.81509602912444354,
     'colsample_bytree': 0.4895702288489546,
     'reg_alpha': 0.791266214098575,
     'min_child_samples': 9,
     'reg_lambda': 3,
}


lgbm_model = LGBMClassifier(
    **lgbm_best_params,
    verbose=-1,
    objective='binary',
    device='gpu' if device == 'cuda' else 'cpu',
    # device=device,
    # max_bin=3600,
    eval_metric='auc',
    random_state=42,
)

lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict_proba(X_test)[:, 1]

print(f'AUC ROC score: {roc_auc_score(y_test, y_pred)}')


# def lgbm_objective(trial):
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 10, 70),
#         'num_leaves': trial.suggest_int('num_leaves', 200, 500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 600),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
#         'subsample': trial.suggest_float('subsample', 0.6, 0.8),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.6),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 4.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
#         'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 0.7),
#     }

#     rocauc = []
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     for train_index, test_index in cv.split(X_train_original, y_train_original):
#         X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
#         y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#         lgbm_model = LGBMClassifier(
#             **params,
#             verbose=-1,
#             objective='binary',
#             device='gpu' if device == 'cuda' else 'cpu',
#             random_state=42,
#         )

#         lgbm_model.fit(
#             X_train_fold, 
#             y_train_fold, 
#             eval_metric='auc',
#             eval_set=[(X_test_fold, y_test_fold)],
#         )
#         y_pred_fold = lgbm_model.predict_proba(X_test_fold)[:, 1]

#         _rocauc = roc_auc_score(y_test_fold, y_pred_fold)
#         rocauc.append(_rocauc)
#     return np.mean(rocauc)

# lgbm_original_study = optuna.create_study(direction='maximize', study_name='lgbm_original_study')
# lgbm_original_study.optimize(lgbm_objective, n_trials=5)
# lgbm_original_best_params = lgbm_original_study.best_params
# print(f"Best parameters: {lgbm_original_best_params}")


# optuna.visualization.plot_optimization_history(lgbm_original_study).show()


# optuna.visualization.plot_param_importances(lgbm_original_study).show()


# optuna.visualization.plot_slice(lgbm_original_study).show()


# lgbm_original_model = LGBMClassifier(
#     **lgbm_original_best_params,
#     verbose=-1,
#     objective='binary',
#     device='gpu' if device == 'cuda' else 'cpu',
#     eval_metric='auc',
#     random_state=42,
# )

# lgbm_original_model.fit(X_train_original, y_train_original)
# y_original_pred = lgbm_original_model.predict_proba(X_test_original)[:, 1]

# print(f'AUC ROC score: {roc_auc_score(y_test_original, y_original_pred)}')


# def catboost_objective(trial):
#     params = {
#         'iterations' : trial.suggest_int('iterations', 2000, 5000),
#         'learning_rate' : trial.suggest_float('learning_rate', 0.03, 0.07),
#         'depth' : trial.suggest_int('depth', 6, 12),
#     }

#     rocauc = []
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     for train_index, test_index in cv.split(X_train, y_train):
#         X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
#         y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#         catboost_model = CatBoostClassifier(
#             **params,
#             eval_metric='AUC',
#             random_seed=42,
#             early_stopping_rounds=50,
#             verbose=0,
#             task_type='GPU' if device == 'cuda' else 'CPU',
#             devices='0'
#         )

#         catboost_model.fit(
#             X_train_fold, 
#             y_train_fold, 
#         )
#         y_pred_fold = catboost_model.predict_proba(X_test_fold)[:, 1]

#         _rocauc = roc_auc_score(y_test_fold, y_pred_fold)
#         rocauc.append(_rocauc)
#     return np.mean(rocauc)

# catboost_study = optuna.create_study(direction='maximize', study_name='catboost_study')
# catboost_study.optimize(catboost_objective, n_trials=5)
# catboost_best_params = catboost_study.best_params
# print(f"Best parameters: {catboost_best_params}")


# optuna.visualization.plot_optimization_history(catboost_study).show()


# optuna.visualization.plot_param_importances(catboost_study).show()


# optuna.visualization.plot_slice(catboost_study).show()


# catboost_best_params = {
#     'iterations': 2998,
#     'learning_rate': 0.03951922827867255,
#     'depth': 6
# }


#  catboost_model = CatBoostClassifier(
#         **catboost_best_params,
#         eval_metric='AUC',
#         random_seed=42,
#         early_stopping_rounds=50,
#         verbose=100,
#         task_type='GPU' if device == 'cuda' else 'CPU',
#         devices='0'
#     )

# catboost_model.fit(X_train, y_train)
# y_pred = catboost_model.predict_proba(X_test)[:, 1]

# print(f'AUC ROC score: {roc_auc_score(y_test, y_pred)}')


lgbm_pred = lgbm_model.predict_proba(X_test)[:, 1]
# lgbm_original_pred = lgbm_original_model.predict_proba(X_test)[:, 1]
# xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
# catboost_pred = catboost_model.predict_proba(X_test)[:, 1]

# test_ensemble = (lgbm_pred + xgb_pred + catboost_pred) / 3
# test_ensemble = (lgbm_pred + xgb_pred) / 2
# test_ensemble = (lgbm_pred + lgbm_original_pred) / 2
print(f'AUC ROC score: {roc_auc_score(y_test, lgbm_pred)}')


errors = y_test - test_ensemble
sns.lineplot(errors)


(errors**2).mean()


X_sub = test.drop('id', axis=1)
X_sub = feat_eng.run(X_sub)
    
X_sub.head()


# y_pred_xgb_sub = xgb_model.predict_proba(X_sub)[:, 1]
y_pred_lgbm_sub = lgbm_model.predict_proba(X_sub)[:, 1]
# y_pred_lgbm_original_sub = lgbm_original_model.predict_proba(X_sub)[:, 1]
# y_pred_catboost_sub = catboost_model.predict_proba(X_sub)[:, 1]
# final_sub = (y_pred_xgb_sub + y_pred_lgbm_sub + y_pred_catboost_sub) / 3
# final_sub = (y_pred_xgb_sub + y_pred_lgbm_sub) / 2
# final_sub = (y_pred_lgbm_sub + y_pred_lgbm_original_sub) / 2


submission = pd.DataFrame({
    'id': test['id'],
    # 'y': final_sub,
    'y' : y_pred_lgbm_sub,
})

submission.to_csv('submission.csv', index=False)


submission.head()




