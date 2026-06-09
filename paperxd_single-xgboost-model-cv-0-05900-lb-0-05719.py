from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import gc

warnings.simplefilter(action = 'ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
org_train = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')
org_train = org_train.rename(columns = {'User_ID':'id', 'Gender':'Sex'})
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

def simple_process(df):
    df = df.drop(columns = ['id'])
    sex_enc = {'female':-1, 'male':1}
    df['Age'] = df['Age'] ** 0.5
    df['Weight'] = 0.4 / df['Weight'] 
    df['Height'] = 0.4 / df['Height'] 
    df['Body_Temp'] = 70 / df['Body_Temp']
    df = df.replace(sex_enc)
    return df

def preprocess(df):
    df.reset_index(drop = True, inplace = True)

    df['BIN_Weight'] = pd.qcut(df['Weight'], q = 5, labels = False)
    df['BIN_Height'] = pd.qcut(df['Height'], q = 5, labels = False)

    values = list(range(5))
    all_perms = list(permutations(values))
    perm_93 = all_perms[93]
    perm_71 = all_perms[71]
    perm_95 = all_perms[95]

    dict_93 = dict(zip(values, perm_93))
    dict_71 = dict(zip(values, perm_71))
    dict_95 = dict(zip(values, perm_95))

    df['BIN_Weight93'] = df.BIN_Weight.replace(dict_93)
    df['BIN_Weight71'] = df.BIN_Weight.replace(dict_71)
    df['BIN_Height93'] = df.BIN_Height.replace(dict_93)
    df['BIN_Height95'] = df.BIN_Height.replace(dict_95)

    for i in values:
        idx_93 = df[df['BIN_Weight93'] == i].index
        df['BIN_Weight93'].iloc[idx_93] = pd.qcut(df[df['BIN_Weight93'] == i].Weight, q = 5, labels = list(range(5*i,5*i+5)))

        idx_71 = df[df['BIN_Weight71'] == i].index
        df['BIN_Weight71'].iloc[idx_71] = pd.qcut(df[df['BIN_Weight71'] == i].Weight, q = 5, labels = list(range(5*i,5*i+5)))

        idx_93 = df[df['BIN_Height93'] == i].index
        df['BIN_Height93'].iloc[idx_93] = pd.qcut(df[df['BIN_Height93'] == i].Height, q = 5, labels = list(range(5*i,5*i+5)))        
        
        idx_95 = df[df['BIN_Height95'] == i].index
        df['BIN_Height95'].iloc[idx_95] = pd.qcut(df[df['BIN_Height95'] == i].Height, q = 5, labels = list(range(5*i,5*i+5))) 

    df['Round_Duration2'] = np.round(df.Duration / 2) * 2
    df['Duration'] = df['Duration'] * 2
    df['Round_Duration2'] = df['Round_Duration2'] * 2
    drop_columns = ['Weight','Height']
    features = ['Sex','Age','Duration','Heart_Rate','Body_Temp','Weight','Height', 'Round_Duration2']
    
    combs_2 = combinations(features, 2)
    combs_3 = combinations(features, 3) 

    df_cols_to_add = []
    for comb in combs_2:
        new_df = pd.DataFrame()
        c1, c2 = comb
        name_mult = '_mult_'.join(comb)
        name_add = '_add_'.join(comb)
        new_df[name_mult] = df[c1] * df[c2]
        new_df[name_add] = df[c1] + df[c2]
        df_cols_to_add.append(new_df)
        del new_df, c1, c2, comb, name_mult
        gc.collect()

    for comb in combs_3:
        new_df = pd.DataFrame()
        c1, c2, c3 = comb
        name_mult = '_mult_'.join(comb)
        name_add = '_add_'.join(comb)
        new_df[name_mult] = df[c1] * df[c2] * df[c3]
        new_df[name_add] = df[c1] + df[c2] + df[c3]
        df_cols_to_add.append(new_df)
        del new_df, c1, c2, comb, name_mult
        gc.collect()
        
    df_add = pd.concat(df_cols_to_add, axis = 1)
    df = pd.concat([df, df_add], axis = 1)
    df = df.drop(columns = drop_columns)
    return df

train = simple_process(train) 
org_train = simple_process(org_train)
train = pd.concat([train, org_train], axis = 0)

x = train.drop('Calories', axis = 1)
y = train.Calories

test = simple_process(test)

params = {'n_estimators': 10000, 'max_depth': 10, 'learning_rate': 0.005, 'subsample': 0.5, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5}

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
oof = np.zeros(len(x))
preds = np.zeros(len(test))
all_importance_types = ['weight','gain','total_gain','cover','total_cover']
importance_all = {imp: defaultdict(list) for imp in all_importance_types}

x = preprocess(x)
test_copy = preprocess(test)
for train_idx, test_idx in kf.split(x,y):
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    model = XGBRegressor(**params, random_state = 42, verbosity = 1, tree_method = 'hist')
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], eval_metric = 'rmse', early_stopping_rounds = 100, verbose = 100)
    oof[test_idx] = np.expm1(model.predict(x_test))
    
    preds += np.expm1(model.predict(test_copy)) / kf.n_splits
    booster = model.get_booster()
    for imp_type in all_importance_types:
        scores = booster.get_score(importance_type = imp_type)
        for feature, value in scores.items():
            importance_all[imp_type][feature].append(value)

    del x_train, x_test, y_train, y_test, model, booster
    gc.collect()

oof = np.clip(oof, 1, 314)
print(np.sqrt(mean_squared_log_error(y, oof)))
np.savetxt('oof.csv',oof,delimiter = ',',fmt = '%.8f')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub.Calories = np.clip(preds, 1, 314)
sub.to_csv('pred.csv', index = False)

features_to_drop = set()

avg_importance = {imp: {f: np.mean(values) for f, values in importance_all[imp].items()} for imp in all_importance_types}
for imp, values in avg_importance.items():
    sorted_values = sorted(values.items(), key = lambda x: x[1], reverse = True)
    sorted_features = [value[0] for value in sorted_values]
    len_values = len(sorted_features)
    perc = 0.95
    perc_values = int(np.round(perc * len_values))
    features_to_drop.update(sorted_features[perc_values:])

    df = pd.DataFrame(sorted_values[:20], columns = ['Feature','Importance'])
    sns.barplot(data = df, x = 'Feature', y = 'Importance')
    plt.xticks(fontsize = 5, rotation = 30)
    plt.title(imp)
    plt.show()

print(features_to_drop)

