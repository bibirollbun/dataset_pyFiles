import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path

plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)


cols_to_drop = {'CustomerId'}
str_cols = {'Surname', 'Geography', 'Gender'}
cast_to_int_cols = {'HasCrCard', 'IsActiveMember'}

def prepare_train(df):
    train_df = df.drop(labels=cols_to_drop | {'id'}, axis=1)
    
    str_col_map_dict = {}
    for str_col in str_cols:
        str_col_map = {cat: i for i, cat in enumerate(train_df[str_col].unique())}
        
        train_df[str_col] = train_df[str_col].map(str_col_map)
        str_col_map_dict[str_col] = str_col_map
    
    for cast_to_int_col in cast_to_int_cols:
        train_df[cast_to_int_col] = train_df[cast_to_int_col].astype(int)
    
    return train_df, str_col_map_dict

def prepare_test(df, str_col_map_dict):
    test_df = df.drop(labels=cols_to_drop, axis=1)
    
    for str_col in str_cols:
        test_df[str_col] = test_df[str_col].map(str_col_map_dict[str_col])
    
    for cast_to_int_col in cast_to_int_cols:
        test_df[cast_to_int_col] = test_df[cast_to_int_col].astype(int)
    
    return test_df

def plot_metric_on_val(values, metric, title):
    x = range(1, len(values) + 1)
    xticks = range(1, len(values) + 1, 5)
    
    plt.figure(figsize=(12, 4))
    plt.plot(x, values, c='c', alpha=0.75, linestyle='--')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.xticks(xticks)
    plt.ylabel(metric)
    plt.grid()
    plt.show()


workdir_path = Path('/kaggle/input/playground-series-s4e1')


raw_train_df = pd.read_csv(workdir_path / 'train.csv')

raw_train_df


info_df = pd.concat(objs=(raw_train_df.dtypes, raw_train_df.isna().sum()), axis=1)
info_df.columns = ('dtype', '#na')

info_df


train_df, str_col_map_dict = prepare_train(raw_train_df)

train_df


target = 'Exited'
features = list(set(train_df.columns) ^ {target})
cat_features = list(str_cols | cast_to_int_cols)

train_ds = lgb.Dataset(train_df[features], label=train_df[target], feature_name=features,
                       categorical_feature=cat_features, free_raw_data=False)

train_ds


params = {
    'objective': 'binary',
    'metric': 'auc'
}

model = lgb.train(params=params, train_set=train_ds, num_boost_round=100)


history = lgb.cv(params=params, train_set=train_ds, num_boost_round=100, nfold=5,
                 callbacks=[lgb.early_stopping(stopping_rounds=5)])


plot_metric_on_val(history['valid auc-mean'], metric='AUC', title='AUC on Validation')


lgb.plot_importance(model, figsize=(12, 4));


lgb.plot_tree(model, figsize=(24, 12));


test_df = prepare_test(pd.read_csv(workdir_path / 'test.csv'), str_col_map_dict)
test_df['Exited'] = model.predict(data=test_df[features],
                                  num_iteration=model.best_iteration)

test_df[['id', 'Exited']].to_csv('submission.csv', index=False)

