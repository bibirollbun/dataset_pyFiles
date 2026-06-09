# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


if 1:
    !pip install -U scikit-learn
    !pip install -U pandas


import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.preprocessing import TargetEncoder  
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')



is_GPU = False

if is_GPU:
    %load_ext cudf.pandas
    device = 'cuda'
    dev_cb = 'GPU'

else:
    device = 'cpu'
    dev_cb = 'CPU'
    



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
debug_fraction = 0.10 # 10%

if DEBUG:    
    train = train.groupby(target, group_keys=False).apply(lambda x: x.sample(frac=debug_fraction, random_state=2025)).reset_index(drop=True)
    test = test.groupby(test.columns[0], group_keys=False).apply(lambda x: x.sample(frac=debug_fraction,random_state=2025)).reset_index(drop=True)  # Random fraction of test data


train.shape


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']


train['balance'] = (train['balance'] // 1000) * 1000
test['balance'] = (test['balance'] // 1000) * 1000
train['duration'] = (train['duration'] // 10) * 10
test['duration'] = (test['duration'] // 10) * 10
train['pdays'] = (train['pdays'] // 10) * 10
test['pdays'] = (test['pdays'] // 10) * 10



train.head(5)


COLS = list( train.columns[0:-1] )
train[COLS] = train[COLS].astype("str")
test[COLS] = test[COLS].astype("str")
print( COLS )
print(len(COLS),"uni-grams exist")


memory_usage_series = train.memory_usage(index=True, deep=True)
total_memory_usage = memory_usage_series.sum()
print(f'Train memory: {total_memory_usage // (1024*1024)} Mb')



if 1:
    new_columns = {}
    new_columns2 = {}
    COLS2 = []
    for i, c1 in enumerate(COLS[:-1]):
        for j, c2 in enumerate(COLS[i+1:]):
            name = f"{c1}-{c2}"
            new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str")
            new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str")
            COLS2.append(name)
            print(f"{i}-{i+j+1}, ", end='')
    train = pd.concat([train, pd.DataFrame(new_columns)], axis=1)
    test = pd.concat([test, pd.DataFrame(new_columns2)], axis=1)
    print()
    print(len(COLS2),"bi-grams generated")


memory_usage_series = train.memory_usage(index=True, deep=True)
total_memory_usage = memory_usage_series.sum()
print(f'Train memory: {total_memory_usage // (1024*1024)} Mb')



train.columns


if 0:
    new_columns = {}
    new_columns2 = {}
    COLS3 = []
    for i, c1 in enumerate(COLS[:-2]):
        for j, c2 in enumerate(COLS[i+1:-1]):
            for k, c3 in enumerate(COLS[i+j+2:]):
                name = f"{c1}-{c2}-{c3}"
                new_columns[name] = train[c1].astype("str") + "_" + train[c2].astype("str") + "_" + train[c3].astype("str")
                new_columns2[name] = test[c1].astype("str") + "_" + test[c2].astype("str") + "_" + test[c3].astype("str")
                COLS3.append(name)
                print(f"{i}-{i+j+1}-{i+j+k+2}, ", end='')
    train = pd.concat([train, pd.DataFrame(new_columns)], axis=1)
    test = pd.concat([test, pd.DataFrame(new_columns2)], axis=1)
    print()
    print(len(COLS3),"tri-grams generated")


memory_usage_series = train.memory_usage(index=True, deep=True)
total_memory_usage = memory_usage_series.sum()
print(f'Train memory: {total_memory_usage // (1024*1024)} Mb')



TARGET = target
#TARGET_ENCODE = [f"{c}-TE" for c in COLS+COLS2+COLS3]
TARGET_ENCODE = [f"{c}-TE" for c in COLS+COLS2]
#TARGET_ENCODE = [f"{c}-TE" for c in COLS]
more_train = pd.DataFrame(data=np.zeros( (len(train),len(TARGET_ENCODE)) ), columns=TARGET_ENCODE)
train = pd.concat([train,more_train],axis=1)
more_test = pd.DataFrame(data=np.zeros( (len(test),len(TARGET_ENCODE)) ), columns=TARGET_ENCODE)
test = pd.concat([test,more_test],axis=1)

FEATURES = TARGET_ENCODE
print(f"Here are all our {len(FEATURES)} features:")
print( FEATURES )


memory_usage_series = train.memory_usage(index=True, deep=True)
total_memory_usage = memory_usage_series.sum()
print(f'Train memory: {total_memory_usage // (1024*1024)} Mb')


memory_usage_series = test.memory_usage(index=True, deep=True)
total_memory_usage = memory_usage_series.sum()
print(f'Train memory: {total_memory_usage // (1024*1024)} Mb')


sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=train[TARGET]
)


FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

# SAVE OOF AND TEST PREDS
oof = np.zeros( len(train) )
pred = np.zeros( len(test) )
roc_scores = []

# TRAIN/INFER K-FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    
    # PRINT FOLD NUMBER
    if i%FOLDS==0: print()
    print("#"*25)
    print(f"### Fold {i+1} ###")
    print("#"*25)
    
    # GET TRAIN, VALID, TEST
    X_train = train.iloc[train_index,].copy()
    y_train = train[TARGET].iloc[train_index]
    
    X_valid = train.iloc[test_index,].copy()
    y_valid = train[TARGET].iloc[test_index]
    
    X_test = test.copy()
    
    # TARGET ENCODE AND STANDARD ENCODE
    print(f"TE for {len(TARGET_ENCODE)} features...")
    for j,c in enumerate(TARGET_ENCODE):
        c = c.replace("-TE","")
        print(f"{j}, ",end="") 
        
        # TARGET ENCODE 
        enc_auto = TargetEncoder(smooth='auto',
                                 random_state=2025,                                  
                                 cv=5)
        X_train[f"{c}-TE"] = enc_auto.fit_transform(X_train[[c]], y_train)
        X_valid[f"{c}-TE"] = enc_auto.transform(X_valid[[c]])
        X_test[f"{c}-TE"] = enc_auto.transform(X_test[[c]])
        
        # STANDARD ENCODE
        m = X_train[f"{c}-TE"].mean()
        s = X_train[f"{c}-TE"].std()
        X_train[f"{c}-TE"] = (X_train[f"{c}-TE"]-m)/s
        X_valid[f"{c}-TE"] = (X_valid[f"{c}-TE"]-m)/s
        X_test[f"{c}-TE"] = (X_test[f"{c}-TE"]-m)/s
    
    print()    
    X_train = X_train[FEATURES]
    X_valid = X_valid[FEATURES]
    X_test = X_test[FEATURES]
            
    # FIT LASSO MODEL
    #model = Lasso(alpha=1e2) 
    model = XGBClassifier(tree_method='hist', 
                                 device=device,                                  
                                 eval_metric='auc',
                                 sample_weight=sample_weights
                                )
    #model = LogisticRegression(solver='saga', penalty='l1') 
    #model = CatBoostClassifier(
    #                            allow_writing_files=False,
    #                            verbose=False,                                
    #                            task_type=dev_cb,
    #                            n_estimators=10000,
    #                            learning_rate=0.03,
    #                        )
    
    model.fit(X_train, y_train)
    
    # INFER OOF AND TEST
    oof[test_index] = model.predict_proba(X_valid)[:, 1]
    if i==0: 
        pred = model.predict_proba(X_test)[:, 1]
    else: 
        pred += model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_valid, oof[test_index])
    
    roc_scores.append(roc_score)
    print(f"Fold {i} -> ROC-AUC: {roc_score:.5f}")

print(f"Average Fold ROC-AUC: {np.mean(roc_scores):.5f} \xb1 {np.std(roc_scores):.5f}")
pred /= FOLDS


if 1:
    # Example data
    data = {'Names': FEATURES,
    #        'Numbers': model.coef_[0, :]}
            'Numbers': model.feature_importances_}

    # Create a DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("Numbers",ascending=True)
    df = df.loc[abs(df.Numbers) > 0.01]

    # Create a horizontal bar plot
    df.plot(x='Names', y='Numbers', kind='barh', legend=False, figsize=(10, 20))

    # Show the plot
    plt.show()


most_relevant = df['Names'].to_list()


most_relevant


if 1:
    sub = pd.read_csv(path_to_data + "sample_submission.csv")
    sub['y'] = pred
    sub.to_csv(path_to_output + "submission.csv", index=False)
    sub.head()




