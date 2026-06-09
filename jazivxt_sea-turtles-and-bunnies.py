import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
from itertools import combinations

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df = pd.DataFrame(train.Personality.value_counts().reset_index())
df.plot(kind='pie', y='count', labels=df['Personality'], figsize=(6, 6))


train.info()


#cols = ['Time_spent_Alone', 'Social_event_attendance',
#       'Going_outside', 'Friends_circle_size',
#       'Post_frequency']
#df = pd.concat((train, test))
#df[cols].median().to_dict()


means = {'Time_spent_Alone': 3,
 'Social_event_attendance': 5,
 'Going_outside': 4,
 'Friends_circle_size': 7,
 'Post_frequency': 4,
 'Stage_fear': 0,
 'Drained_after_socializing': 0}
das = {'No': 1, 'Yes': -1}
sf = {'No': 1, 'Yes': -1}
target = {'Extrovert': 0, 'Introvert': 1}
rtarget = {0:'Extrovert', 1:'Introvert'}


cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']
for c in cols:
    train[c+'isnull'] = train[c].isnull().astype(int) - 0.5
    test[c+'isnull'] = test[c].isnull().astype(int) - 0.5
train['nullcount'] = train.isnull().sum(axis=1)
test['nullcount'] = test.isnull().sum(axis=1)


train['Stage_fear'] = train['Stage_fear'].map(sf)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(das)
train['Personality'] = train['Personality'].map(target)


test['Stage_fear'] = train['Stage_fear'].map(sf)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(das)



cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency', ] #'Time_spent_Aloneisnull',
       #'Stage_fearisnull', 'Social_event_attendanceisnull',
       #'Going_outsideisnull', 'Drained_after_socializingisnull',
       #'Friends_circle_sizeisnull', 'Post_frequencyisnull', 'nullcount'


simputes = ['Friends_circle_size', 'Drained_after_socializing', 'Social_event_attendance', 'Time_spent_Alone', 'Post_frequency', 'Going_outside', 'Stage_fear']
for simp in simputes:
    scols = [c for c in cols if c != simp] #+ ['Personality']
    model = ensemble.ExtraTreesRegressor(n_estimators=1000, max_depth=9, n_jobs=-1, random_state=2)
    x1 = train[train[simp].isnull()].reset_index(drop=True)
    x2 = train[train[simp].notnull()]

    T1 = test[test[simp].isnull()].reset_index(drop=True)
    T2 = test[test[simp].notnull()].reset_index(drop=True)

    x2temp = pd.concat([x2, T2]).dropna().reset_index(drop=True)
    
    model.fit(x2temp[scols], x2temp[simp])
    x1[simp] = model.predict(x1[scols].fillna(means))
    train = pd.concat([x2, x1]).reset_index(drop=True)

    T1[simp] = model.predict(T1[scols].fillna(means))
    test = pd.concat([T2, T1]).reset_index(drop=True)


train = train.fillna(means)
test = test.fillna(means)


pf = preprocessing.PolynomialFeatures(degree=2)
sc = preprocessing.StandardScaler()
X = sc.fit_transform(pf.fit_transform(train[cols]))
Y = train['Personality']
T = sc.transform(pf.transform(test[cols]))

params = {
    'objective': 'multi:softprob',
    'num_class': 2,
    'eval_metric': 'mlogloss',
    'max_depth': 19,
    'learning_rate': 0.002,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'tree_method': 'hist', 
    #'device': 'cuda',
    'random_state': 42
}

skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
oof_preds = np.zeros((len(X), 2))
preds = np.zeros((len(T), 2))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(f"\nFold {fold+1}")
    dtrain = xgb.DMatrix(X[trn_idx], label=Y[trn_idx])
    dvalid = xgb.DMatrix(X[val_idx], label=Y[val_idx])

    model = xgb.train(params, dtrain, num_boost_round=10_000,
                      evals=[(dvalid, 'valid')],
                      early_stopping_rounds=50,
                      verbose_eval=100)
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    preds += model.predict(xgb.DMatrix(T), iteration_range=(0, model.best_iteration)) / skf.n_splits


threshold = 0.5
score = 0.0
for i in np.arange(0.2, 0.8, 0.01):
    train['target'] = (oof_preds[:,1] > i).astype(int)
    s = np.sum((train['target'] == train['Personality']).astype(int)) / len(train)
    if s > score:
        score = s
        threshold = i
print(threshold, score)


test['Personality'] = (preds[:,1] > threshold).astype(int)
test['Personality'] = test['Personality'].map(rtarget)
test[['id','Personality']].to_csv('submission.csv', index=False)

