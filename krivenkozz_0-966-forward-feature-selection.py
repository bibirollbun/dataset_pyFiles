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


if 0:
    !pip install -U scikit-learn
    !pip install -U pandas


import numpy as np
import pandas as pd
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression,Lasso
#from sklearn.preprocessing import TargetEncoder  
from matplotlib import pyplot as plt
from sklearn.base import clone
import json
import time

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)



is_GPU = True

if is_GPU:
    %load_ext cudf.pandas
    device = 'cuda'    

else:
    device = 'cpu'


path_to_data = '/kaggle/input/playground-series-s5e8/'
path_to_output = '/kaggle/working/'


train = pd.read_csv(path_to_data + "train.csv") 
print("Train shape:", train.shape )
train.head(5)


test = pd.read_csv(path_to_data + "test.csv")
print("Test shape:", test.shape)
test.head(5)


target = 'y'
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


DEBUG = False
debug_fraction = 0.1 # 10%

if DEBUG:    
    train = train.groupby(target, group_keys=False).apply(lambda x: x.sample(frac=debug_fraction, random_state=2025)).reset_index(drop=True)
    test = test.groupby(test.columns[0], group_keys=False).apply(lambda x: x.sample(frac=debug_fraction,random_state=2025)).reset_index(drop=True)  # Random fraction of test data


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']


train['balance'] = (train['balance'] // 1000) * 1000
test['balance'] = (test['balance'] // 1000) * 1000
train['duration'] = (train['duration'] // 10) * 10
test['duration'] = (test['duration'] // 10) * 10
train['pdays'] = (train['pdays'] // 10) * 10
test['pdays'] = (test['pdays'] // 10) * 10


RMV = ["id","y"]
FEATURES = [c for c in train.columns if not c in RMV]
combined = pd.concat([train,test], axis=0, ignore_index=True)

CATS = []
HIGH_CARDINALITY = []
print(f"THE {len(FEATURES)} BASIC FEATURES ARE:")

for c in FEATURES:
    ftype = "numerical"
    if combined[c].dtype=="object":
        CATS.append(c)
        combined[c] = combined[c].fillna("NAN")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        ftype = "categorical"
    if combined[c].dtype=="int64":
        combined[c] = combined[c].astype("int32")
    elif combined[c].dtype=="float64":
        combined[c] = combined[c].astype("float32")
        
    n = combined[c].nunique()
    print(f"{c} ({ftype}) with {n} unique values")
    if n>=9: HIGH_CARDINALITY.append(c)
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

print("\nTHE FOLLOWING HAVE 9 OR MORE UNIQUE VALUES:", HIGH_CARDINALITY )


def target_encode(train, valid, test, col, target="y", kfold=5, smooth=20, agg="mean"):

    train['kfold'] = ((train.index) % kfold)
    col_name = '_'.join(col)
    train[f'TE_{agg.upper()}_' + col_name] = 0.
    for i in range(kfold):       
        df_tmp = train[train['kfold']!=i]
        if agg=="mean": mn = train[target].mean()
        elif agg=="median": mn = train[target].median()
        elif agg=="min": mn = train[target].min()
        elif agg=="max": mn = train[target].max()
        elif agg=="nunique": mn = 0        
        df_tmp = df_tmp[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
        df_tmp.columns = col + [agg, 'count']
        if agg=="nunique":
            df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
        else:
            df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
        df_tmp_m = train[col + ['kfold', f'TE_{agg.upper()}_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        df_tmp_m.loc[df_tmp_m['kfold']==i, f'TE_{agg.upper()}_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_tmp']
        train[f'TE_{agg.upper()}_' + col_name] = df_tmp_m[f'TE_{agg.upper()}_' + col_name].fillna(mn).values  
    
    df_tmp = train[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
    if agg=="mean": mn = train[target].mean()
    elif agg=="median": mn = train[target].median()
    elif agg=="min": mn = train[target].min()
    elif agg=="max": mn = train[target].max()
    elif agg=="nunique": mn = 0
    df_tmp.columns = col + [agg, 'count']
    if agg=="nunique":
        df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
    else:
        df_tmp['TE_tmp'] = ((df_tmp[agg]*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
    
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_{agg.upper()}_' + col_name] = valid[f'TE_{agg.upper()}_' + col_name].astype("float32")

    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_{agg.upper()}_' + col_name] = test[f'TE_{agg.upper()}_' + col_name].astype("float32")

    train = train.drop('kfold', axis=1)
    train[f'TE_{agg.upper()}_' + col_name] = train[f'TE_{agg.upper()}_' + col_name].astype("float32")

    return(train, valid, test)


def count_encode(train, valid, test, combined, col, target="y"):
# COUNT ENCODING (USING COMBINED TRAIN TEST)
    tmp = combined.groupby(col)[target].count()
    nm = f"CE_{'_'.join(col)}"; tmp.name = nm
    train = train.merge(tmp, on=col, how="left")
    valid = valid.merge(tmp, on=col, how="left")
    test = test.merge(tmp, on=col, how="left")
    train[nm] = train[nm].astype("int32")
    valid[nm] = valid[nm].astype("int32")
    test[nm] = test[nm].astype("int32")
    return(train, valid, test)


FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2025)
train_index, test_index = next(kf.split(train[FEATURES], train[target]))


x_train = train.loc[train_index,FEATURES+["y"] ].copy()
y_train = train.loc[train_index,"y"]
x_valid = train.loc[test_index,FEATURES].copy()
y_valid = train.loc[test_index,"y"]
x_test = test[FEATURES].copy()


xgb_clf = XGBClassifier(tree_method='hist', 
                                device=device,                                  
                                eval_metric='auc',
                                random_state=2025
                                )


for col in FEATURES:
    x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, [col], smooth=20, agg="mean")
    x_train, x_valid, x_test = count_encode(x_train, x_valid, x_test, combined, [col], target="y")


y_to_add = x_train[target]
x_train = x_train.drop("y",axis=1)


model = clone(xgb_clf)
model.fit(x_train, y_train, 
              eval_set=[(x_valid, y_valid)], 
              early_stopping_rounds=200, 
              verbose=1000
              )
    
y_pred = model.predict_proba(x_valid)[:, 1]
best_score = roc_auc_score(y_valid, y_pred)
print(f"Basic ROC-AUC: {best_score:.5f}")


MORE = []


start_time = time.time()
for k in range(101):
    ct = np.random.choice([2,3,4,5,6],1)[0]    
    new =  list( np.random.choice(FEATURES, ct, replace=False))
    x_train = pd.concat((x_train, y_to_add), axis=1)
    x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, new, smooth=20, agg="mean")
    #x_train, x_valid, x_test = count_encode(x_train, x_valid, x_test, combined, new, target)
    x_train = x_train.drop("y",axis=1)
    
    model = clone(xgb_clf)
    model.fit(x_train, y_train, 
              eval_set=[(x_valid, y_valid)], 
              early_stopping_rounds=200, 
              verbose=0
              )
    
    y_pred = model.predict_proba(x_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred)
    #print(f"ROC-AUC: {score:.5f}")

    if k%50 == 0:
        middle_time = time.time()
        print(f'k: {k}, time elapsed: {(middle_time - start_time):.0f}')

    if score > best_score:
        MORE.append(new)
        best_score = score
        print(f"ROC-AUC: {best_score:.5f}, MORE length: {len(MORE)}")
    else:
        n = "_".join(new)
        #rmv = [f"TE_MEAN_{n}", f"CE_{n}"]
        rmv = [f"TE_MEAN_{n}"]
        x_train = x_train.drop(rmv,axis=1)
        x_valid = x_valid.drop(rmv, axis=1)

print(f"Final ROC-AUC: {best_score:.5f}, Final MORE length: {len(MORE)}")


with open(path_to_output + 'MORE.json', 'w') as file:
    json.dump(MORE, file, indent=4) # indent for pretty printing


%%time

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2025)

oof = np.zeros(len(train))
pred = np.zeros(len(test))
roc_scores =[]

for i, (train_index, test_index) in enumerate(kf.split(train[FEATURES], train[target])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES+["y"] ].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    start = time.time()
    print(f"FEATURE ENGINEER {len(FEATURES)} COLUMNS and {len(MORE)} GROUPS: ",end="")
    for j,f in enumerate(FEATURES+MORE):

        if j<len(FEATURES): c = [f]
        else: c = f 
        print(f"({j+1}){c}",", ",end="")

        # LOW CARDINALITY FEATURES - TARGET ENCODE MEAN AND MEDIAN
        x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=20, agg="mean")
        x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="median")

        # HIGH CARDINALITY FEATURES - TE MIN, MAX, NUNIQUE and CE
        if (j>=len(FEATURES)) | (c[0] in HIGH_CARDINALITY):
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="min")
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="max")
            x_train, x_valid, x_test = target_encode(x_train, x_valid, x_test, c, smooth=0, agg="nunique")
    
            # COUNT ENCODING (USING COMBINED TRAIN TEST)
            tmp = combined.groupby(c).y.count()
            nm = f"CE_{'_'.join(c)}"; tmp.name = nm
            x_train = x_train.merge(tmp, on=c, how="left")
            x_valid = x_valid.merge(tmp, on=c, how="left")
            x_test = x_test.merge(tmp, on=c, how="left")
            x_train[nm] = x_train[nm].astype("int32")
            x_valid[nm] = x_valid[nm].astype("int32")
            x_test[nm] = x_test[nm].astype("int32")
            
    end = time.time()
    elapsed = end-start
    print(f"Feature engineering took {elapsed:.1f} seconds")
    x_train = x_train.drop("y",axis=1)

    model = clone(xgb_clf)
    model.fit(x_train, y_train, 
              eval_set=[(x_valid, y_valid)], 
              early_stopping_rounds=200, 
              verbose=100
              )

    # INFER OOF
    oof[test_index] = model.predict_proba(x_valid)[:, 1]
    # INFER TEST
    pred += model.predict_proba(x_test)[:, 1]
    
    m = roc_auc_score(y_valid, oof[test_index])
    roc_scores.append(m)
    print(f"Fold {i+1}, ROC-AUC: {m:.5f}")
    
print(f"Average Fold ROC-AUC: {np.mean(roc_scores):.5f} \xb1 {np.std(roc_scores):.5f}")
# COMPUTE AVERAGE TEST PREDS
pred /= FOLDS


m = roc_auc_score(train.y.values, oof)


fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size if needed
plot_importance(
    model,
    ax=ax,
    max_num_features=25,  # Display only the top 25 features
    importance_type="weight",  # Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
)
plt.title("XGB Top 25 Feature Importances")
plt.show()


sub = pd.read_csv(path_to_data + "sample_submission.csv")
sub['y'] = pred
sub.to_csv(path_to_output + "submission.csv", index=False)
sub.head()




