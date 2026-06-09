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


seed=42


import random
import torch
import numpy as np


os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, Lasso
import optuna
from sklearn.tree import DecisionTreeRegressor



train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')



drop_cols=['id']


train=train.drop(columns=drop_cols)
test=test.drop(columns=drop_cols)





num_feats=train.select_dtypes(include='number').columns.tolist()
cat_feats=train.select_dtypes(include='object').columns.tolist() # их нету



train.head()


train.info()


for col in num_feats:
    plt.figure()
    plt.hist(train[col])
    plt.xlabel(col)
    plt.ylabel('частота')
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

for col in num_feats:
    plt.figure(figsize=(5,3))
    sns.histplot(train[col].dropna(), bins='auto', kde=True, stat='density')
    plt.title(col)
    plt.tight_layout()
    plt.show()



goal=train.corrwith(train['BeatsPerMinute']) # мщжно методы разные пробовать 
plt.figure()
goal.plot(kind='bar')
plt.show()





goal


goal


def featurize(df):
    #InstrumentalScore   и   VocalContent
    df['log_InstrumentalScore']=np.log1p(df['InstrumentalScore'].astype(float))
    df['log_VocalContent']=np.log1p(df['VocalContent'].astype(float))
    df['new_feat1']= (df['AudioLoudness'] * df['RhythmScore']) / df['TrackDurationMs']

    df['new_feat2']= df['VocalContent'] ** df['InstrumentalScore']
    df['new_feat3']= df['Energy'] + df['MoodScore'] 
    df['new_feat4']= df['TrackDurationMs']* df['RhythmScore'] -(df['AcousticQuality'] + df['InstrumentalScore'])
    df['new_feat5']=df['MoodScore']- df['RhythmScore']**df['TrackDurationMs']
    df['new_feat6']=(df['LivePerformanceLikelihood']+df['VocalContent']+df['AcousticQuality'])/100

    #new
    df['new_feat7']=(df['MoodScore']+df['log_VocalContent'])/df['RhythmScore']
    df['new_feat8']=df['VocalContent']*df['log_InstrumentalScore'] + df['MoodScore']

    #new new

    df['new_feat9']=df['MoodScore']/df['Energy']+df['AudioLoudness']-df['AcousticQuality']
    df['new_feat10']=df['AudioLoudness']+df['InstrumentalScore']-df['log_InstrumentalScore']
    df['new_feat11']=df['TrackDurationMs']-df['VocalContent']-df['MoodScore']

    df['new_feat12']=df['new_feat7']/df['new_feat5']
    return df


train_new=featurize(train)
test_new=featurize(test)


train_new


trying=train_new.corrwith(train_new['BeatsPerMinute'])


trying


X=train_new.drop(columns='BeatsPerMinute')
y=train_new['BeatsPerMinute']


y = np.log1p(y)            # ln(1 + y)
def inv_log(pred):
    return np.expm1(pred) 


#C = float(max(y.max(), 1.0))    # большая константа; можно зафиксировать, напр. C=1000.0
#y_scaled = y / C
#def inv_scale(pred):
 #   return pred * C


model_cfgs = [
    dict(name="depth6_lr0.03",
         params=dict(iterations=2000, learning_rate=0.03, depth=6, loss_function='RMSE',
                     l2_leaf_reg=3.0, task_type='GPU', devices='0', random_seed=42)),
    dict(name="cb_depth8_lr0.02",
         params=dict(iterations=3000, learning_rate=0.02, depth=8, loss_function='RMSE',
                     l2_leaf_reg=6.0,  task_type='GPU', devices='0', random_seed=42)),
    dict(name="depth10_lr0.015",
         params=dict(iterations=4000, learning_rate=0.015, depth=10, loss_function='RMSE',
                     l2_leaf_reg=8.0, bagging_temperature=0.5, task_type='GPU', devices='0', random_seed=42)),
    dict(name="growlossols_lr0.03",
         params=dict(iterations=2500, learning_rate=0.03, depth=7, loss_function='RMSE',
                     grow_policy='Lossguide', l2_leaf_reg=4.0, task_type='GPU', devices='0', random_seed=42)),
    dict(name="border64_lr0.025",
         params=dict(iterations=2500, learning_rate=0.025, depth=6, loss_function='RMSE',
                     border_count=64, l2_leaf_reg=5.0, task_type='GPU', devices='0', random_seed=42)),
]


kf=KFold(n_splits=5, shuffle=True,  random_state=42)
n_models = len(model_cfgs)
oof_preds  = np.zeros((len(y), n_models), dtype=float)
test_preds = np.zeros((len(test_new),  n_models), dtype=float)

cv_scores  = []

for m_idx, cfg in enumerate(model_cfgs):
    name, params=cfg['name'], cfg['params']
    oof_pred=np.zeros(len(y), dtype=float)
    test_pred=np.zeros(len(test_new), dtype=float)
    fold_rmse = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X,y), start=1):
        X_train, X_val=X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val=y.iloc[tr_idx], y.iloc[val_idx]
        train_pool=Pool(
        data=X_train,
        label=y_train
        )
        val_pool=Pool(
        data=X_val,
        label=y_val
        )
        test_pool=Pool(
        data=test_new
        )
        model=CatBoostRegressor(
        **params
        
        )
        model.fit(train_pool, eval_set=val_pool, verbose=200, use_best_model=True, early_stopping_rounds=400)
        val_pred = model.predict(val_pool)
        oof_pred[val_idx] = val_pred
        fold_rmse.append(mean_squared_error(y_val, val_pred, squared=False))

        test_pred += model.predict(test_pool) / 5
    print(f'CV RMSE: {np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}')
    oof_preds[:, m_idx]  = oof_pred
    test_preds[:, m_idx] = test_pred
    model_cv = (np.mean(fold_rmse), np.std(fold_rmse))
    cv_scores.append((name, *model_cv))
    print(f"{name}: CV RMSE {model_cv[0]:.5f} ± {model_cv[1]:.5f}")




colnames = [cfg["name"] for cfg in model_cfgs]
oof_df   = pd.DataFrame(oof_preds,  columns=colnames)
test_df  = pd.DataFrame(test_preds, columns=colnames)


model_lasso=Lasso(alpha=1)



model_lasso.fit(oof_df, y)

linreg_test_pred = model_lasso.predict(test_df)



sub1=pd.DataFrame({
    'id': sample['id'],
    'BeatsPerMinute' : linreg_test_pred
})


sub1.to_csv('sub_linear_reg2.csv', index=False)


model_cat=CatBoostRegressor(
    iterations=500,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=10.0,
    task_type='GPU',
    devices='0',
    loss_function='RMSE',
    
)


model_cat.fit(oof_df, y, verbose=False)

catboost_test_pred=np.expm1(model_cat.predict(test_df))



sub2=pd.DataFrame({
    'id': sample['id'],
    'BeatsPerMinute': catboost_test_pred
})


sub2.to_csv('sub_catboost.csv', index=False)


model_tree=DecisionTreeRegressor(
    max_depth=3, 
    min_samples_leaf=100,
    random_state=42
)


model_tree.fit(oof_df, y )
tree_test_pred=model_tree.predict(test_df)



sub3=pd.DataFrame({
    'id': sample['id'],
    'BeatsPerMinute': tree_test_pred
})


sub3.to_csv('sub_decision_tree.csv', index=False)








def objective(trial):
     raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n_models)])
     w = raw / (raw.sum() + 1e-12)
     blend = oof_df.values @ w
     return mean_squared_error(y, blend, squared=False)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)
best_raw = np.array([study.best_params[f"w{i}"] for i in range(n_models)])
w = best_raw / (best_raw.sum() + 1e-12)
print("Best weights:", dict(zip(colnames, w.round(4))))
test_blend = np.expm1(test_df.values @ w)

#Своровал но понял что происходит, надо самим реализовать в след ноутбуке


sub4=pd.DataFrame({
    'id': sample['id'],
    'BeatsPerMinute': test_blend
})


sub4.to_csv('sub_optuna.csv', index=False)








sub_total=pd.DataFrame({
    'id': sample['id'],
    'BeatsPerMinute': 0.4* sub4['BeatsPerMinute'] + 0.6 * sub2['BeatsPerMinute']
})


sub_total.to_csv('sub_catboost_plus_optuna_opt.csv', index=False)




