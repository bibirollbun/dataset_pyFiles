import random

import os

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import matplotlib.pyplot as plt
import seaborn as sns
import graphviz



N = 250*1000

N_FOLDS = 5
N_TRIALS = 25



seed = 1723

random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)









x, y = make_classification(n_samples=10000, weights=[.80, .20])

feats_space = [f'feat{i}' for i in range(x.shape[1])]
df = pd.DataFrame(x, columns=feats_space)
df['label'] = y

train, valid = train_test_split(df)

train_dataset = lgb.Dataset(train[feats_space], 
                            label=train['label'], 
                            init_score=np.zeros(len(train)))
valid_dataset = lgb.Dataset(valid[feats_space], 
                            label=valid['label'], 
                            init_score=np.zeros(len(valid)))



class NewLoss:
    
    def __init__(self):
        self.iteration = 0
        
    def __call__(self, y_pred, d_train):

        if self.iteration == 0:
            print (y_pred[:10])

        y_true = d_train.get_label()
        sigmoid_pred = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -250, 250)))
        
        gradients = sigmoid_pred - y_true
        hessians = sigmoid_pred * (1.0 - sigmoid_pred)
        
        if self.iteration < 3:
            print (self.iteration, len(gradients), len(set(gradients)), gradients[:5])
            print (self.iteration, len(hessians), len(set(hessians)), hessians[:5])

        self.iteration += 1
        
        return gradients, hessians

def new_metrics(y_pred, d_train):
    
    y_true = d_train.get_label()
    
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -250, 250)))
    p += 1e-15
    
    metrics = -np.mean(y_true * np.log(p) + (1-y_true) * np.log(1-p))
    
    return 'logloss', metrics, False  



params = {
    'objective': NewLoss(), 
    "max_depth": 5, 
    "verbose": -1 
}
evals_result = {}
model = lgb.train(params, 
                  feval=new_metrics,  
                  num_boost_round=100,   
                  train_set=train_dataset, 
                  valid_sets=[train_dataset, 
                              valid_dataset], 
                  callbacks=[lgb.record_evaluation(evals_result)])



mean = df['label'].mean()
bias = np.log((mean) / (1-mean))

mean, bias



for i in [0, 9]:
    graph_data = lgb.create_tree_digraph(model, tree_index=i)
    graph_data.render("tree")
    display(graph_data)



l = [
    (evals_result['training']['logloss'], 'training (all rounds)', 'blue'), 
    (evals_result['training']['logloss'][-50:], 'training (last 50)', 'blue'), 
    
    (evals_result['valid_1']['logloss'], 'validation (all rounds)', 'red'), 
    (evals_result['valid_1']['logloss'][-50:], 'validation (last 50)', 'red')
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (data, title, color) in enumerate(l):
    sns.lineplot(data, color=color, ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel('rounds')
    axes[i].set_ylabel('loss')

plt.tight_layout()
plt.show()









## data_colection

df = pd.read_csv("../input/microsoft-malware-prediction/train.csv", 
                 nrows=N)

submission = pd.read_csv("../input/microsoft-malware-prediction/test.csv", 
                         nrows=N)

df.head()



## data_split

df, test = train_test_split(df, 
                            test_size=.20, 
                            random_state=1723)
df.reset_index(drop=True, 
               inplace=True)
test.reset_index(drop=True, 
                 inplace=True)

inds = []
kf = KFold(n_splits=N_FOLDS, 
           shuffle=True, 
           random_state=1723)
for (train_index, valid_index) in kf.split(df):
    inds += [[train_index, valid_index]]



## feats_engin 

target = "HasDetections"
nums = [
    "Census_ProcessorCoreCount", 
    "Census_PrimaryDiskTotalCapacity", 
    "Census_SystemVolumeTotalCapacity", 
    "Census_TotalPhysicalRAM", 
    "Census_InternalPrimaryDisplayResolutionHorizontal", 
    "Census_InternalPrimaryDisplayResolutionVertical", 
    "Census_InternalPrimaryDiagonalDisplaySizeInInches", 
    "Census_InternalBatteryNumberOfCharges"
]
cats = [c for c in df.columns if c not in nums + [target, 'MachineIdentifier']]

for col in cats:
    
    uni = set(df[df[col].isnull() == False][col])
    d = dict(
        zip(uni, 
            range(len(uni)))
    )
    
    df[col] = df[col].map(d)
    test[col] = test[col].map(d)    
    submission[col] = submission[col].map(d)









## feats_selection  

def training_phase(train, valid, 
                   nums, cats, 
                   target, 
                   cs={}):

    train_dataset = lgb.Dataset(train[nums+cats], 
                                train[target], 
                                categorical_feature=cats)
    valid_dataset = lgb.Dataset(valid[nums+cats], 
                                valid[target], 
                                categorical_feature=cats)
    
    params = {
        "objective": "binary", 
        "boosting_type": "gbdt", 
        "num_boost_rounds": 1000, 
        "early_stopping_round": 100, 
        "max_depth": -1, 
        "bagging_fraction": .5, 
        "features_fraction": .5, 
        "verbose": -1, 
        "nthreads": 4, 
        "random_state": 1723
    }
    params.update(cs)
    model = lgb.train(params=params, 
                      train_set=train_dataset, 
                      valid_sets=valid_dataset)
    
    esti = model.predict(valid[nums+cats])
    ms = roc_auc_score(valid[target], 
                       esti)
    
    return model, ms

df["label"] = 1
submission["label"] = 0

fr = pd.concat([df, submission], 
               axis=0)
fr.reset_index(drop=True, 
               inplace=True)

train, valid = train_test_split(fr, 
                                test_size=.20, 
                                random_state=1723)
train.reset_index(drop=True, 
                  inplace=True)
valid.reset_index(drop=True, 
                  inplace=True)

shift = []
tr = 1

while tr:
    
    model, metrics = training_phase(train, valid, 
                                    nums, cats, 
                                    "label")
    
    if metrics > .667:
        
        ind = np.argmax(model.feature_importance("gain"))
        features_space = nums+cats
        print (features_space[ind], 
               metrics)
        
        cats = [
            c for c in cats if c != features_space[ind]
        ]
        nums = [
            c for c in nums if c != features_space[ind]
        ]
        shift += [features_space[ind]]
    
    else:
        tr = 0

df.drop('label', 
        axis=1, 
        inplace=True)
submission.drop('label', 
                axis=1, 
                inplace=True)



## feats_selection  

def cats_exploration(df, shift):
    
    df = df[[shift, 'MachineIdentifier']].groupby(shift).count().reset_index()
    df.columns = [shift, 'statistics']
    
    df['statistics'] /= df['statistics'].sum()
    df.sort_values('statistics', 
                   ascending=False, 
                   inplace=True)
    df.reset_index(drop=True, 
                   inplace=True)

    print (df.head())

cats_exploration(df, shift[0])
cats_exploration(submission, shift[0])









## feats_selection  

def null_importance(train, valid, 
                    nums, cats, 
                    target):
    
    model, metrics = training_phase(train, valid, 
                                    nums, cats, 
                                    target)
    
    splits = [list(model.feature_importance(importance_type='split'))]
    gains = [list(model.feature_importance(importance_type='gain'))]

    for i in tqdm(range(128)):

        tra = train.copy()
        tra[target] = np.random.permutation(tra[target].values)
        
        val = valid.copy()
        val[target] = np.random.permutation(val[target].values)

        model, metrics = training_phase(tra, val, 
                                        nums, cats, 
                                        target)
        
        splits += [list(model.feature_importance(importance_type='split'))]
        gains += [list(model.feature_importance(importance_type='gain'))]

    splits = pd.DataFrame(data=splits, columns=nums+cats)
    gains = pd.DataFrame(data=gains, columns=nums+cats)

    return splits, gains

splits, gains = null_importance(df.iloc[inds[0][0]], 
                                df.iloc[inds[0][1]], 
                                nums, cats, 
                                target)

splits.head(16)



## feats_selection  

gains.head(16)



## feats_selection  

for col in gains.columns:
    if np.max(gains[col][1:]) > gains[col][0] / 10: 
        
        plt.figure(figsize=(20, 3))
        
        sns.distplot(
            gains[col][1:], 
            kde=False, 
            hist_kws={
                'linewidth': 2,  
                'color': 'red'  
            }
        )
        
        plt.axvline(
            gains[col][0], 
            color='blue', 
            linewidth=5
        )
        
        plt.show()









splits, gains = [], []
for fold in tqdm(range(N_FOLDS)):
    model, metrics = training_phase(df.iloc[inds[fold][0]], 
                                    df.iloc[inds[fold][1]], 
                                    nums, cats, 
                                    target)
    
    splits += [list(model.feature_importance(importance_type='split'))]
    gains += [list(model.feature_importance(importance_type='gain'))]

splits = pd.DataFrame(data=splits, columns=nums+cats)
gains = pd.DataFrame(data=gains, columns=nums+cats)

gains



gains = gains.rank(axis=1)

gains



splits = splits.rank(axis=1)

splits









## model_trainer 

def optuna_objective(trial: Trial):
    
    params = {
        "boosting_type": ["cat", ["gbdt", "dart"]], 
        "max_depth": ["int", 2, 14], 
        "learning_rate": ["float", 0.005, 1], 
        "bagging_fraction": ["float", 0.25, 1], 
        "feature_fraction": ["float", 0.25, 1], 
        "lambda_l1": ["float", 0.1, 2], 
        "lambda_l2": ["float", 0.1, 2]  
    }
    
    cs = dict()
    for k in params:
        v = params[k]
        if v[0] == "int":
            cs[k] = trial.suggest_int(k, v[1], v[2])
        elif v[0] == "float":
            cs[k] = trial.suggest_float(k, v[1], v[2])
        else:
            cs[k] = trial.suggest_categorical(k, v[1])

    metrics = []
    for fold in range(N_FOLDS):
        
        train = df.iloc[inds[fold][0]]
        train.reset_index(drop=True, 
                          inplace=True)
        
        valid = df.iloc[inds[fold][1]]
        valid.reset_index(drop=True, 
                          inplace=True)
        
        metrics += [
            training_phase(train, valid, 
                           nums, cats, 
                           target, 
                           cs)[1]
        ]
    
    return np.mean(metrics) - np.std(metrics)

sampler = TPESampler(seed=1723)
study = optuna.create_study(sampler=sampler, 
                            direction="maximize")
study.optimize(optuna_objective, N_TRIALS)

stats = study.trials_dataframe()
stats.sort_values('value', 
                  ascending=False, 
                  inplace=True)
stats.reset_index(drop=True, 
                  inplace=True)

stats.head()



## competition  

artefacts = []
for fold in tqdm(range(N_FOLDS)):
    
    train = df.iloc[inds[fold][0]]
    train.reset_index(drop=True, 
                      inplace=True)
    
    valid = df.iloc[inds[fold][1]]
    valid.reset_index(drop=True, 
                      inplace=True)
    
    artefacts += [
        training_phase(train, valid, 
                       nums, cats, 
                       target, 
                       study.best_params)
    ]

    model = artefacts[fold][0]
    if fold == 0:
        test['esti'] = model.predict(test[nums+cats])
        submission[target] = model.predict(submission[nums+cats])
    else:
        test['esti'] += model.predict(test[nums+cats])
        submission[target] += model.predict(submission[nums+cats])

print (
    [np.round(art[1], 5) for art in artefacts], 
    np.round(roc_auc_score(test[target], 
                           test['esti']), 5), 
    '\n'
)

submission = submission[['MachineIdentifier', target]].copy()
submission.to_csv('submission.csv', 
                  index=False)

submission.head()











