import warnings
warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
train.shape, test.shape, orig.shape


# Create new features from the original datas
TARGET = 'diagnosed_diabetes'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]
print(f'{len(BASE)} Base Features:{BASE}')

ORIG = []
for col in BASE:
    if col == 'id':
        continue
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()  # return a Series
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name) # .size return a Array, reset_index make a DF
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(len(ORIG), 'Orig Features Created!!')

train.shape, test.shape


# Label Encode
for col in CATS:
    lbl = LabelEncoder()
    lbl.fit(list(train[col]) + list(test[col]))
    train[col] = lbl.transform(train[col])
    test[col] = lbl.transform(test[col])


X = train.drop(['diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
from sklearn.metrics import accuracy_score,roc_auc_score

space={ 'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 1000,
        'seed': 0,
        'eval_metric': 'auc',
        'early_stopping_rounds':200,
        'verbose':False,
        'enable_categorical':True,
        'tree_method': 'gpu_hist',           # 确保使用 GPU 训练
        'predictor': 'gpu_predictor'         # 启用 GPU 加速预测 (可选，但推荐)
    }

def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']),
                    enable_categorical=True,                    
                    device='cuda' # 可以保留此参数，与 tree_method='gpu_hist' 协同工作
                    )
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation,
            verbose = 0, 
            )
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


trials = Trials()
best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)


best_hyperparams 


best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'] )


# 标准off训练法
import xgboost as xgb
import gc
kf = StratifiedKFold(n_splits=5)

oof = np.zeros(len(X))
preds = np.zeros(len(test))

for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"KFold = {i+1}")
    #clf = xgb.XGBClassifier(**best_hyperparams,n_estimators=1000,subsample=0.8,tree_method= 'hist', device='cuda')
    clf = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=11,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.6042463243373577,
            gamma = 1.0140305758721704,
            min_child_weight = 3.0,
            reg_alpha = 52,
            reg_lambda = 0.6037726084089807,
            enable_categorical=True,
            missing=-1,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=100, 
        device='cuda'
       ) 

    
    h = clf.fit(X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx],y.iloc[val_idx])],
            verbose=200)
    
    oof[val_idx] += clf.predict_proba(X.iloc[val_idx])[:,1]
    preds += clf.predict_proba(test)[:,1] / kf.n_splits
    del h, clf
    x = gc.collect()

print('#'*30)
print ('OOF CV=',roc_auc_score(y, oof))


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sample_submission.diagnosed_diabetes = preds
sample_submission.to_csv('baseline-my.csv',index=False)




