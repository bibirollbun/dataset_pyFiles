!pip install lifelines -q --no-index --find-links=/kaggle/input/cibmtr2024-import/lifelines


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostRanker
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from catboost import Pool
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier


import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))




data_dir = '/kaggle/input/equity-post-HCT-survival-predictions'
random_state = 69

df_dict = pd.read_csv(os.path.join(data_dir, 'data_dictionary.csv'))
df_dict.head()


df_dict.info()


df_dict['type'].tolist()


df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_train = df_train.drop(columns=['ID'])
df_train


df_test


columns_train = set(df_train.columns)
columns_test = set(df_test.columns)
target_columns = list(columns_train - columns_test)
target_columns


for column in target_columns:
    print(df_train[column].unique().max())


df_train.isna().sum()


df_test.isna().sum()


cat_features = df_dict[df_dict['type'] == 'Categorical']['variable'].tolist()
numerical_features = list(columns_train - set(cat_features))
cat_features.pop(cat_features.index('efs'))
len(cat_features)


def reduce_nans(df, cat_features):
                
    for column in tqdm(list(df.columns)):
        if column in cat_features:
            df[column] = df[column].fillna("unknown")

    for column in tqdm(list(df.columns)):
        if column not in cat_features:
            x_train = df_train.loc[df_train[column].isna() == False, cat_features]
            y_train = df_train.loc[df_train[column].isna() == False, 'efs']

            cat_imputer = CatBoostRegressor(verbose=0, iterations=600)
            cat_imputer.fit(x_train, y_train, cat_features=cat_features)
            
            x_test = df_train.loc[df_train[column].isna() == True, cat_features]
            y_pred = cat_imputer.predict(x_test)

            df_train.loc[df_train[column].isna() == True, column] = y_pred

    return df


def reduce_nans_simple(df, cat_features):

    for column in tqdm(list(df.columns)):
        if column in cat_features:
            df[column + '_nan'] = 0

    df['cnt_nan'] = 0
    
    for row in tqdm(range(len(df))):
        cnt = 0
        for i, column in enumerate(df.columns):
            if df.iloc[row, i] == 'NaN':
                df.loc[row, column + '_nan'] = 1
                cnt += 1
        df.loc[row, 'cnt_nan'] = cnt
    
    for column in tqdm(list(df.columns)):
        df[column] = df[column].fillna("unknown" if column in cat_features else -1)

    return df
    

def apply_mean_target_encoding(df, df_test, target_column, columns, cat_features):
    
    for column in tqdm(columns):
        if column == 'ID':
            continue

        if column not in cat_features:
            continue
        unique_values = df[column].unique()
        
        df[column + '_num'] = 0.0
        df[column + '_prob'] = 0.0
        df_test[column + '_num'] = 0.0
        df_test[column + '_prob'] = 0.0
        
        for uniq in unique_values:
            mean_target = df[(df[column] == uniq) & (df['efs'] == 1.0)][target_column].mean()
            mean_prob = df[df[column] == uniq]['efs'].mean()
            df.loc[df[column] == uniq, column + '_num'] = mean_target
            df.loc[df[column] == uniq, column + '_prob'] = mean_prob
            df_test.loc[df_test[column] == uniq, column + '_num'] = mean_target
            df_test.loc[df_test[column] == uniq, column + '_prob'] = mean_prob
            
    cat_feat = [cf for cf in cat_features if cf != 'race_group']
    df = df.drop(columns=cat_feat)
    df_test = df_test.drop(columns=cat_feat)
    return df, df_test



nans_table = df_train.isna().sum()
without_nans_cols = [column for column in df_train.columns 
                     if column != 'ID' and nans_table[column] == 0 and column not in target_columns]

# df_train = reduce_nans(df_train, cat_features, without_nans_cols)
df_train = reduce_nans_simple(df_train, cat_features)
df_test = reduce_nans_simple(df_test, cat_features)
df_train, df_test = apply_mean_target_encoding(df_train, df_test, 'efs_time', list(df_train.columns), cat_features)

df_train.isna().sum().sum() == 0


df_train.head()


sns.heatmap(df_train.corr(numeric_only=True), annot=False)


naf = NelsonAalenFitter()
        
naf.fit(df_train['efs_time'], df_train['efs'])
df_train['naf_label'] = -naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_train.loc[df_train['efs'] == 0, 'naf_label'] -= 0.1
        
kmf = KaplanMeierFitter()
kmf.fit(df_train['efs_time'], df_train['efs'])
df_train['km_label'] = kmf.survival_function_at_times(df_train['efs_time']).values
df_train.loc[df_train['efs'] == 0, 'km_label'] -= 0.1
df_train


df_train['cox'] = df_train['efs_time'] * df_train['efs'].apply(lambda x: 1 if x == 1 else -1)
df_train


'race_group' in df_train.columns and 'race_group' in cat_features


df_test['race_group'].unique()


from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from joblib import dump

les_dir = 'label_encoders'
if not Path(les_dir).exists():
    os.mkdir(les_dir)
    
for column in df_train.columns:
    if column == 'race_group':
        le = LabelEncoder()
        le.fit(df_train[column].tolist())
        df_train[column] = le.transform(df_train[column])
        df_test[column] = le.transform(df_test[column])
        



df_train['y_lower'] = df_train['efs_time']
df_train['y_upper'] = np.where(df_train['efs'], df_train['efs_time'], df_train['efs_time'].max() + 10)


x = df_train.drop(columns=target_columns + ['y_lower', 'y_upper', 'naf_label', 'km_label', 'cox'])
y = df_train[target_columns + ['y_lower', 'y_upper', 'naf_label', 'km_label', 'cox']]


# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(2, interaction_only=True)

# input_features = list(x.columns)
# xx = poly.fit_transform(x)
# all_features_names = poly.get_feature_names_out(input_features=input_features)
# x = pd.DataFrame(xx, columns=all_features_names)
# x 


# x['race_group'] = x['race_group'].astype(np.int32)
x['race_group'].unique()


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
main_target = 'km_label'

def objective(trial):

    # train_data = Pool(data=x_train, label=y_train[['y_lower', 'y_upper']], cat_features=cat_features)
    # test_data = Pool(data=x_test, label=y_test[['y_lower', 'y_upper']], cat_features=cat_features)
    train_data = Pool(data=x_train, label=y_train[main_target], cat_features=['race_group'])
    test_data = Pool(data=x_test, label=y_test[main_target], cat_features=['race_group'])
    
    param = {
        # "task_type": 'GPU',
        # 'devices': '0',
        "iterations": trial.suggest_int("iterations", 400, 1200),
        # "loss_function": trial.suggest_categorical("loss_function", ["SurvivalAft"]),
        "loss_function": trial.suggest_categorical("loss_function", ["RMSE"]),
        
        # "loss_function": "Cox",
        # "eval_metric": trial.suggest_categorical("eval_metric", ['SurvivalAft']),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 4, 50),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat = CatBoostRegressor(**param)
    # cat = CatBoostClassifier(**param)
    

    cat.fit(train_data, eval_set=test_data, verbose=0, early_stopping_rounds=100)

    pred = cat.predict(x_test)
    preds = pred
    y_pred = pd.DataFrame(columns=['id', 'prediction'])
    y_pred['id'] = list(range(len(preds)))
    y_pred['prediction'] = preds

    y_true = pd.DataFrame(columns=['id', 'efs', 'efs_time', 'race_group'])
    y_true['id'] = list(range(len(preds)))
    y_true['efs'] = y_test['efs'].tolist()
    y_true['efs_time'] = y_test['efs_time'].tolist()
    y_true['race_group'] = [x_test['race_group'].tolist()[i] for i in range(len(preds))]
    # y_true['race_group'] = 'race_group1'

    metricc = score(y_true.copy(), y_pred.copy(), 'id')
    return metricc


def objective_xgb(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    xgb = XGBRegressor(**param)
    # cat = CatBoostClassifier(**param)
    

    xgb.fit(x_train, y_train[main_target], eval_set=[(x_test, y_test[main_target])], verbose=0)

    pred = xgb.predict(x_test)
    preds = pred
    y_pred = pd.DataFrame(columns=['id', 'prediction'])
    y_pred['id'] = list(range(len(preds)))
    y_pred['prediction'] = preds

    y_true = pd.DataFrame(columns=['id', 'efs', 'efs_time', 'race_group'])
    y_true['id'] = list(range(len(preds)))
    y_true['efs'] = y_test['efs'].tolist()
    y_true['efs_time'] = y_test['efs_time'].tolist()
    y_true['race_group'] = [x_test['race_group'].tolist()[i] for i in range(len(preds))]
    # y_true['race_group'] = 'race_group1'

    metricc = score(y_true.copy(), y_pred.copy(), 'id')
    return metricc

def objective_lgbm(trial):
    param = {
        'device': 'gpu',
        'objective': 'mae',
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'verbose': -1,
    }

    lgbm = LGBMRegressor(**param)
    # cat = CatBoostClassifier(**param)
    

    lgbm.fit(x_train, y_train[main_target], eval_set=[(x_test, y_test[main_target])])

    pred = lgbm.predict(x_test)
    preds = pred
    y_pred = pd.DataFrame(columns=['id', 'prediction'])
    y_pred['id'] = list(range(len(preds)))
    y_pred['prediction'] = preds

    y_true = pd.DataFrame(columns=['id', 'efs', 'efs_time', 'race_group'])
    y_true['id'] = list(range(len(preds)))
    y_true['efs'] = y_test['efs'].tolist()
    y_true['efs_time'] = y_test['efs_time'].tolist()
    y_true['race_group'] = [x_test['race_group'].tolist()[i] for i in range(len(preds))]
    # y_true['race_group'] = 'race_group1'

    metricc = score(y_true.copy(), y_pred.copy(), 'id')
    return metricc


from joblib import dump

def train_and_eval(model_name='lightgbm', target='km_label', n_trials=10):

    main_target = target
    match model_name:
        case 'catboost': obj = objective
        case 'xgboost': obj = objective_xgb
        case 'lightgbm': obj = objective_lgbm
    
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, n_jobs=4)
    trial = study.best_trial
    
    if model_name == 'catboost':
        model = CatBoostRegressor(**(trial.params))
        
        train_data = Pool(data=x_train, label=y_train[main_target], cat_features=['race_group'])
        test_data = Pool(data=x_test, label=y_test[main_target], cat_features=['race_group'])
        
        model.fit(train_data, eval_set=test_data, verbose=200, early_stopping_rounds=100)
        
    elif model_name == 'xgboost':
        model = XGBRegressor(**(trial.params))
        model.fit(x_train, y_train[main_target], eval_set=[(x_test, y_test[main_target])], verbose=200)

    elif model_name == 'lightgbm':
        model = LGBMRegressor(**(trial.params))
    
        model.fit(x_train, y_train[main_target], eval_set=[(x_test, y_test[main_target])])
    
    pred = model.predict(x_test)
        
    preds = pred
    y_pred = pd.DataFrame(columns=['id', 'prediction'])
    y_pred['id'] = list(range(len(preds)))
    y_pred['prediction'] = preds
        
    y_true = pd.DataFrame(columns=['id', 'efs', 'efs_time', 'race_group'])
    y_true['id'] = list(range(len(y_pred)))
    y_true['efs'] = y_test['efs'].tolist()
    y_true['efs_time'] = y_test['efs_time'].tolist()
    y_true['race_group'] = [x_test['race_group'].tolist()[i] for i in range(len(y_pred))]
    cv_score = score(y_true.copy(), y_pred.copy(), 'id')
    print(cv_score)
    return model, cv_score
                


model, _ = train_and_eval('catboost', 'km_label', 5)


top_k = 50
feat_importance = model.get_feature_importance()
# feat_importance = model.feature_importances_

priorities = []
for i, column in enumerate(x.columns):
    priorities.append([feat_importance[i], column])

priorities.sort(reverse=True)
priorities[:top_k]


from sklearn.metrics import accuracy_score

classifier = CatBoostClassifier(iterations=2000, learning_rate=0.005)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
classifier.fit(x_train, y_train['efs'], eval_set=(x_test, y_test['efs']), use_best_model=True, verbose=100)
accuracy_score(classifier.predict(x_test), y_test['efs'])


# ranker = CatBoostRanker(
#     iterations=30000,
#     task_type='GPU',
#     devices='0',
#     loss_function='YetiRankPairwise',
#     # loss_function='QueryRMSE',
#     eval_metric='NDCG'
# )

# y['cox'] -= y['cox'].min()

# xx, yy = [], []

# for row in tqdm(range(len(x))):
#     if y.loc[row, 'efs'] == 1:
#         xx.append(x.iloc[row, :].to_numpy())
#         yy.append(y.iloc[row, :].to_numpy())

# new_x = pd.DataFrame(xx, columns=x.columns)
# new_y = pd.DataFrame(yy, columns=y.columns)

# x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, test_size=0.2, random_state=random_state)

# x_train = x_train.reset_index(drop=True)
# x_test = x_test.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)

# indexes0_train = [row for row in range(len(y_train)) if y_train.loc[row, 'efs'] == 0]
# indexes1_train = [row for row in range(len(y_train)) if y_train.loc[row, 'efs'] == 1]
# indexes0_test = [row for row in range(len(y_test)) if y_train.loc[row, 'efs'] == 0]
# indexes1_test = [row for row in range(len(y_test)) if y_train.loc[row, 'efs'] == 1]

# x_train = pd.concat([x_train.iloc[indexes0_train, :], x_train.iloc[indexes1_train, :]])
# x_test = pd.concat([x_test.iloc[indexes0_test, :], x_test.iloc[indexes1_test, :]])
# y_train = pd.concat([y_train.iloc[indexes0_train, :], y_train.iloc[indexes1_train, :]])
# y_test = pd.concat([y_test.iloc[indexes0_test, :], y_test.iloc[indexes1_test, :]])

# group_id_train = [0 if row < len(indexes0_train) else 1 for row in range(len(y_train))]
# group_id_test = [0 if row < len(indexes0_test) else 1 for row in range(len(y_test))]

# train_data = Pool(
#     data=x_train,
#     label=y_train['km_label'],
#     group_id=[0] * len(y_train)
# )

# test_data = Pool(
#     data=x_test,
#     label=y_test['km_label'],
#     group_id=[0] * len(y_test)
# )

# ranker.fit(train_data, eval_set=test_data, verbose=500, early_stopping_rounds=4000)
# preds = ranker.predict(x_test)
# y_pred = pd.DataFrame(columns=['id', 'prediction'])
# y_pred['id'] = list(range(len(preds)))
# y_pred['prediction'] = preds
        
# y_true = pd.DataFrame(columns=['id', 'efs', 'efs_time', 'race_group'])
# y_true['id'] = list(range(len(y_pred)))
# y_true['efs'] = y_test['efs'].tolist()
# y_true['efs_time'] = y_test['efs_time'].tolist()
# y_true['race_group'] = [x_test['race_group'].tolist()[i] for i in range(len(y_pred))]
# cv_score = score(y_true.copy(), y_pred.copy(), 'id')
# print(cv_score)


# pred = ranker.predict(df_test.drop(columns=['ID']))
# submission = pd.DataFrame(columns=['ID', 'prediction'])
# submission['ID'] = df_test['ID']
# submission['prediction'] = pred
# submission


# submission.to_csv('submission.csv', index=False)


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

models_lgbm = []
for i, (train_indexes, test_indexes) in enumerate(skf.split(df_train.drop(columns=['race_group']), df_train['race_group'])):
    print(f'fold: {i + 1}')
    x_train, x_test, y_train, y_test = x.iloc[train_indexes, :], x.iloc[test_indexes, :], y.iloc[train_indexes, :], y.iloc[test_indexes, :]
    model_lgbm_km, cv_score = train_and_eval('lightgbm', 'km_label', 2)
    models_lgbm.append([model_lgbm_km, cv_score])


models_cat = []
for i, (train_indexes, test_indexes) in enumerate(skf.split(df_train.drop(columns=['race_group']), df_train['race_group'])):
    print(f'fold: {i + 1}')
    x_train, x_test, y_train, y_test = x.iloc[train_indexes, :], x.iloc[test_indexes, :], y.iloc[train_indexes, :], y.iloc[test_indexes, :]
    model_cat_km, cv_score = train_and_eval('catboost', 'km_label', 5)
    models_cat.append([model_lgbm_km, cv_score])





models_lgbm_naf = []
for i, (train_indexes, test_indexes) in enumerate(skf.split(df_train.drop(columns=['race_group']), df_train['race_group'])):
    print(f'fold: {i + 1}')
    x_train, x_test, y_train, y_test = x.iloc[train_indexes, :], x.iloc[test_indexes, :], y.iloc[train_indexes, :], y.iloc[test_indexes, :]
    model_lgbm_naf, cv_score = train_and_eval('lightgbm', 'naf_label', 5)
    models_lgbm_naf.append([model_lgbm_naf, cv_score])


models_cat_naf = []
for i, (train_indexes, test_indexes) in enumerate(skf.split(df_train.drop(columns=['race_group']), df_train['race_group'])):
    print(f'fold: {i + 1}')
    x_train, x_test, y_train, y_test = x.iloc[train_indexes, :], x.iloc[test_indexes, :], y.iloc[train_indexes, :], y.iloc[test_indexes, :]
    model_cat_naf, cv_score = train_and_eval('catboost', 'naf_label', 5)
    models_cat_naf.append([model_lgbm_naf, cv_score])


print('lightgbm km: ')
for i, (model, cv_score) in enumerate(models_lgbm):
    print(f'fold {i}: {cv_score}')



print('lightgbm naf: ')
for i, (model, cv_score) in enumerate(models_lgbm_naf):
    print(f'fold {i}: {cv_score}')


print('catboost km: ')
for i, (model, cv_score) in enumerate(models_cat):
    print(f'fold {i}: {cv_score}')


print('catboost naf: ')
for i, (model, cv_score) in enumerate(models_cat_naf):
    print(f'fold {i}: {cv_score}')


pred = np.zeros((len(df_test)))
# best_cat_model = None
# best_lgbm_model = None
# best_cv_score_cat = 0.0
# best_cv_score_lgbm = 0.0

# for model, cv_score in models_cat:
#     if best_cv_score_cat <= cv_score:
#         best_cat_model, best_cv_score_cat = model, cv_score

# for model, cv_score in models_lgbm:
#     if best_cv_score_lgbm <= cv_score:
#         best_lgbm_model, best_cv_score_lgbm = model, cv_score
        
for model, cv_score in models_cat:
    cat_pred = model.predict(df_test.drop(columns=['ID']))
    pred += cat_pred
for model, cv_score in models_lgbm:
    lgbm_pred = model.predict(df_test.drop(columns=['ID']))
    pred += lgbm_pred

for model, cv_score in models_cat_naf:
    cat_pred = model.predict(df_test.drop(columns=['ID']))
    pred += cat_pred
for model, cv_score in models_lgbm_naf:
    lgbm_pred = model.predict(df_test.drop(columns=['ID']))
    pred += lgbm_pred
# pred += best_cat_model.predict(df_test.drop(columns=['ID']))
# pred += best_lgbm_model.predict(df_test.drop(columns=['ID']))
    
pred /= 10
submission = pd.DataFrame(columns=['ID', 'prediction'])
submission['ID'] = df_test['ID']
submission['prediction'] = pred
submission


submission.to_csv('submission.csv', index=False)














































