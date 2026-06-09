import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

import xgboost as xgb
from xgboost import XGBClassifier
import optuna

from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv').set_index('id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv').set_index('id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
orig_datasert = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
orig_dataset = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


df_train = pd.concat([df_train, orig_datasert, orig_dataset], axis=0, ignore_index=True)


sum_nan  = df_train.isna().sum(axis=1)
sum_nan[sum_nan>0].size


missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100

missing_train


#df_train.dropna(inplace=True, ignore_index=True)


le = LabelEncoder()
df_train['Personality'] = le.fit_transform(df_train['Personality'])

y = df_train['Personality']
df_train = df_train.drop(['Personality'], axis=1)


# Combine for consistent encoding
combined = pd.concat([df_train, df_test], axis=0)

categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numeric_cols = combined.drop(columns=categorical_cols).columns

# Impute
#num_imputer = SimpleImputer(strategy='median')
#combined[numeric_cols] = num_imputer.fit_transform(combined[numeric_cols])

#cat_imputer = SimpleImputer(strategy='most_frequent')
#combined[categorical_cols] = cat_imputer.fit_transform(combined[categorical_cols])

# encode
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
combined[categorical_cols] = oe.fit_transform(combined[categorical_cols])
combined[categorical_cols] = combined[categorical_cols].astype("category")

# One-hot encoding
#combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

 # Scale
scaler = StandardScaler()
combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

# combined['Social_event_bin'] = pd.qcut(
#     combined['Social_event_attendance'],
#     q=[0, 0.25, 0.5, 0.75, 1.0],
#     labels=['Q1', 'Q2', 'Q3', 'Q4']
# )
#combined['Time_spent_Alone'].fillna(combined.groupby('Social_event_bin')['Time_spent_Alone'].transform('median'), inplace=True)
#combined.drop(columns=['Social_event_bin'], inplace=True)

# Split back
X = combined.iloc[:len(df_train)]
x_test = combined.iloc[len(df_train):]

combined.head(20)


train = pd.concat([X, y], axis=1)
corr_train = train.corr()
mask_train = np.triu(np.ones_like(corr_train, dtype=bool), k=1)

plt.figure(figsize=(8,8))
sns.set_style('white')
sns.heatmap(
    data=corr_train,
    annot=True,
    fmt='.4f',
    mask=mask_train,
    square=True,
    cmap='coolwarm',
    annot_kws={'size':8},
    cbar=False
)
plt.tight_layout()
plt.show()


#mutual_info = mutual_info_regression(X.fillna(0), y, random_state=42)
mutual_info = mutual_info_classif(X.fillna(0), y, random_state=56)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])
mutual_info.style.bar(subset=['Mutual Information'], cmap='RdYlGn')


X.info()


train_preds = {'xgb':np.zeros((len(X), 2)), 'lgb':np.zeros((len(X), 2))}
test_preds = {'xgb':np.zeros((len(x_test), 2)), 'lgb':np.zeros((len(x_test), 2))}


x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=36, stratify=y)


xgb_threshold = 0.5
def xgb_custom_accuracy_score(labels:np.ndarray, predt:np.ndarray):
    predt = [1 if pt > xgb_threshold else 0 for pt in predt]
    accuracy = accuracy_score(labels, predt)
    return accuracy


# Define objective function for Optuna
def xgb_objective(trial):
    
    threshold = trial.suggest_float('threshold', 0.5, 1, step=0.001)
    def xgb_accuracy_score(labels:np.ndarray, predt:np.ndarray):
        predt = [1 if pt > threshold else 0 for pt in predt]
        accuracy = accuracy_score(labels, predt)
        return accuracy

    early_stop = xgb.callback.EarlyStopping(rounds=50,
                                        metric_name='xgb_accuracy_score',
                                        data_name='validation_1',
                                        maximize=True)
    # Hyperparameters to tune
    xgb_params = {
        'objective': 'binary:logistic',
        'n_estimators':3000,
        'device': 'cuda',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'max_bin': trial.suggest_int('max_bin', 80, 150),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.5, 10),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.5, 10),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.1, 1),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1),
        'random_state': 42,
        'enable_categorical':True,
        'eval_metric': xgb_accuracy_score,
#        'early_stopping_rounds':50,
        'callbacks':[early_stop]
    }

    xbg_model = XGBClassifier(**xgb_params)
    xbg_model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train),(x_valid, y_valid)], 
        verbose=10
    )

    best_score = xbg_model.best_score
    # y_pred = xbg_model.predict(x_valid)
    # print(y_pred)
    # accuracy = accuracy_score(y_valid, y_pred)
    return best_score
    
# 创建研究对象
study = optuna.create_study(direction='maximize')  # 目标是最大化准确率
study.optimize(xgb_objective, n_trials=50)  # 进行 50 次试验

# 输出最佳结果
print(f'XGBoost Best trial: {study.best_trial.value}')
print(f'XGBoost Best parameters: {study.best_trial.params}')


xgb_studied_params = study.best_trial.params
xgb_threshold = xgb_studied_params.pop('threshold')

xgb_scores = []
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=78)
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    x_train_split, x_valid_split = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_split, y_valid_split = y.iloc[train_idx], y.iloc[valid_idx]

    early_stop = xgb.callback.EarlyStopping(rounds=50,
                                        metric_name='xgb_custom_accuracy_score',
                                        data_name='validation_0',
                                        maximize=True)
    xgb_params = {
        'objective': 'binary:logistic',
        'n_estimators':3000,
        'device': 'cuda',  
        'tree_method': 'hist',
        'random_state': 42,
        'eval_metric': xgb_custom_accuracy_score,
    #    'early_stopping_rounds':50,
        **xgb_studied_params,
        'callbacks':[early_stop],
        'enable_categorical':True,
    }
    xbg_model = XGBClassifier(**xgb_params)
    xbg_model.fit(
        x_train_split,
        y_train_split,
        eval_set=[(x_train_split, y_train_split), (x_valid_split, y_valid_split)], 
        verbose=10
    )

    train_preds['xgb'][valid_idx] = xbg_model.predict_proba(x_valid_split)
    test_preds['xgb'] += xbg_model.predict_proba(x_test) / FOLDS

    best_score = xbg_model.best_score
    xgb_scores.append(best_score)
    print(f'XGBoost Fold {i+1} score：{best_score:.5f}')
print(f'XGBoost average score：{np.mean(xgb_scores):.5f}')


lgb_threshold = 0.5
def lgb_custom_accuracy_score(labels:np.ndarray, predt:np.ndarray):
    predt = [1 if pt > lgb_threshold else 0 for pt in predt]
    accuracy = accuracy_score(labels, predt)
    return 'accuracy_score', accuracy, True


# Define objective function for Optuna
def lgb_objective(trial):
    
    threshold = trial.suggest_float('threshold', 0.5, 1, step=0.001)
    def lgb_accuracy_score(labels:np.ndarray, predt:np.ndarray):
        predt = [1 if pt > threshold else 0 for pt in predt]
        accuracy = accuracy_score(labels, predt)
        return 'accuracy_score', accuracy, True

    lgb_params = {
        'objective':'binary',
        'boosting_type':'gbdt',
        'data_sample_strategy':'goss',
        'device':'gpu',
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'colsample_bynode':trial.suggest_uniform('colsample_bynode', 0.1, 1),
        'max_bin':trial.suggest_int('max_bin', 80, 150),
        'max_depth':trial.suggest_int('max_depth', 5, 30),
        'min_child_samples':trial.suggest_int('min_child_samples', 10, 200),
        'min_child_weight':trial.suggest_float('min_child_weight', 1, 10),
        'n_estimators':500,
        'bagging_freq':500,
        'baggin_fraction':trial.suggest_uniform('baggin_fraction', 0.1, 1),
        'num_leaves':trial.suggest_int('num_leaves', 10, 200),
        'scale_pos_weight':trial.suggest_uniform('scale_pos_weight', 0, 10),
        'n_jobs':-1,
        'reg_alpha':trial.suggest_uniform('reg_alpha', 0.5, 10),
        'reg_lambda':trial.suggest_uniform('reg_lambda', 0.5, 10),
        'random_state':42,
        'categorical_feature':categorical_cols,
        'early_stopping_round':50,
        'verbose':-1
    }
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=lgb_accuracy_score
    )

    best_score = lgb_model.best_score_['valid_0']['accuracy_score']
    return best_score
    
# 创建研究对象
study = optuna.create_study(direction='maximize')  # 目标是最大化准确率
study.optimize(lgb_objective, n_trials=50)  # 进行 50 次试验

# 输出最佳结果
print(f'LGBMClassifier Best trial: {study.best_trial.value}')
print(f'LGBMClassifier Best parameters: {study.best_trial.params}')


lgb_studied_params = study.best_trial.params
lgb_threshold = lgb_studied_params.pop('threshold')
#lightGBM
lgb_params = {
    'objective':'binary',
    'boosting_type':'gbdt',
    'data_sample_strategy':'goss',
    'device':'gpu',
    **lgb_studied_params,
    'n_estimators':500,
    'bagging_freq':500,
    'n_jobs':-1,
    'random_state':42,
    'categorical_feature':categorical_cols,
    'early_stopping_round':50,
    'verbose':-1
}
lgb_model = LGBMClassifier(**lgb_params)

lgb_scores = []
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=89)
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    x_train_split, x_valid_split = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_split, y_valid_split = y.iloc[train_idx], y.iloc[valid_idx]

    lgb_model.fit(
        x_train_split,
        y_train_split,
        eval_set=[(x_valid_split, y_valid_split)],
        eval_metric=lgb_custom_accuracy_score
    )

    train_preds['lgb'][valid_idx] = lgb_model.predict_proba(x_valid_split)
    test_preds['lgb'] += lgb_model.predict_proba(x_test) / FOLDS

    best_score = lgb_model.best_score_['valid_0']['accuracy_score']
    lgb_scores.append(best_score)
    print(f'LightGBM Fold {i+1} score：{best_score:.5f}')
print(f'LightGBM average score：{np.mean(lgb_scores):.5f}')


stacking_train_preds = np.hstack([train_preds[name] for name in train_preds])
stacking_test_preds = np.hstack([test_preds[name] for name in test_preds])

x_train, x_valid, y_train, y_valid = train_test_split(stacking_train_preds, y, test_size=0.2, random_state=23, stratify=y)


def lr_objective(trial):
    solver_penalty_options = [
        ('liblinear', 'l1'),
        ('liblinear', 'l2'),
        ('lbfgs', 'l2'),
        ('lbfgs', None),
        ('newton-cg', 'l2'),
        ('newton-cg', None),
        ('newton-cholesky', 'l2'),
        ('newton-cholesky', None)
    ]
    solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_options)
    
    lr_params = {
        'random_state': 42,
        'max_iter': 500,
        'C': trial.suggest_float('C', 0, 10),
        'tol': trial.suggest_float('tol', 1e-10, 1e-1),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'solver': solver,
        'penalty': penalty
    }

    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(x_train, y_train)

    lr_preds = lr_model.predict(x_valid)
    lr_score = accuracy_score(y_valid, lr_preds)

    return lr_score

# 创建研究对象
study = optuna.create_study(direction='maximize')  # 目标是最大化准确率
study.optimize(lr_objective, n_trials=50)  # 进行 50 次试验

# 输出最佳结果
print(f'LogisticRegression Best trial: {study.best_trial.value}')
print(f'LogisticRegression Best parameters: {study.best_trial.params}')


lr_studied_params = study.best_trial.params
solver, penalty = lr_studied_params.pop('solver_penalty')
lr_params = {
    'max_iter':500,
    **lr_studied_params,
    'solver':solver,
    'penalty':penalty,
    'random_state':42
}
lr_model = LogisticRegression(**lr_params)

lr_scores = []
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=61)
for i, (train_idx, valid_idx) in enumerate(kf.split(stacking_train_preds, y)):
    x_train_split, x_valid_split = stacking_train_preds[train_idx], stacking_train_preds[valid_idx]
    y_train_split, y_valid_split = y.iloc[train_idx], y.iloc[valid_idx]
    
    lr_model.fit(x_train_split, y_train_split)

    lr_preds_split = lr_model.predict(x_valid_split)
    lr_score = accuracy_score(y_valid_split, lr_preds_split)
    lr_scores.append(lr_score)
    print(f'LogisticRegression FOLD {i+1} score：{lr_score:.5f}')
print(f'LogisticRegression average score：{np.mean(lr_scores):.5f}')



# xgb_params = {
#     'objective': 'binary:logistic',
#     'n_estimators':3000,
#     'device': 'cuda',  
#     'tree_method': 'hist',  
#     'max_depth': 22,
#     'max_bin': 100,
#     'learning_rate': 0.03,
# #    'reg_alpha': 0.8,
# #    'reg_lambda': 4.0,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'colsample_bylevel': 0.7,  
#     'colsample_bynode': 1,
#     'random_state': 42,
#     'enable_categorical':True,
#     'eval_metric': xgb_custom_accuracy_score,
#     'early_stopping_rounds':50,
# }
# xbg_model = XGBClassifier(**xgb_params)

# xbg_model.fit(
#     x_train,
#     y_train,
#     eval_set=[(x_train, y_train),(x_valid, y_valid)], 
#     verbose=10
# )

# print(f'best_iteration：{xbg_model.best_iteration}')
# print(f'best score：{xbg_model.best_score}')


xgb_validation_result = xbg_model.evals_result()

validation_logloss = pd.DataFrame({
    'train_logloss': xgb_validation_result['validation_0']['logloss'],
    'test_logloss': xgb_validation_result['validation_1']['logloss'],
    'xgb_train_accuracy_score': xgb_validation_result['validation_0']['xgb_custom_accuracy_score'],
    'xgb_test_accuracy_score': xgb_validation_result['validation_1']['xgb_custom_accuracy_score'],
})
validation_logloss.index.name = 'trees'

plt.figure(figsize=(10, 6))
sns.lineplot(data=validation_logloss)
plt.title("validation logloss")
plt.tight_layout()
plt.show()


# FOLDS = 5
# kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# xgb_params = {
#     'objective': 'binary:logistic',
#     'n_estimators':3000,
#     'device': 'cuda',  
#     'tree_method': 'hist',  
#     'max_depth': 10,
#     'learning_rate': 0.03,
#     'colsample_bytree': 1,
#     'subsample': 0.8,
#     'colsample_bylevel': 1,  
#     'colsample_bynode': 1,
#     'random_state': 42,
#     'enable_categorical':True,
#     'eval_metric': 'logloss',
#     'early_stopping_rounds':50,
# }
# xbg_model = XGBClassifier(**xgb_params)

# all_logloss = []
# for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
#     x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
#     y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
#     xbg_model.fit(
#         x_train,
#         y_train,
#         eval_set=[(x_valid, y_valid)], 
#         verbose=49
#     )

#     all_logloss.append(xbg_model.best_score)
# print(f"Average logloss across {FOLDS} folds: {np.mean(all_logloss):.4f}")


#xgb_test_pred = xbg_model.predict(x_test)
lgb_test_pred = lgb_model.predict(x_test)
#lr_test_preds = lr_model.predict(stacking_test_preds)
test_labels = le.inverse_transform(lgb_test_pred)
submission = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': test_labels
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


importances = xbg_model.feature_importances_
feature_names = xbg_model.feature_names_in_

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

