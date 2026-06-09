# This Python 3 environment comes with many helpful analytics libraries installed

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


if 1:
    !pip install -U feature_engine==1.5.1


import numpy as np
import pandas as pd
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.preprocessing import LabelEncoder
from feature_engine.encoding import WoEEncoder, RareLabelEncoder, PRatioEncoder
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
debug_fraction = 0.01 # 1% 

if DEBUG:    
    train = train.groupby(target, group_keys=False).apply(lambda x: x.sample(frac=debug_fraction, random_state=2025)).reset_index(drop=True)
    test = test.groupby(test.columns[0], group_keys=False).apply(lambda x: x.sample(frac=debug_fraction,random_state=2025)).reset_index(drop=True)  # Random fraction of test data


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']


le_factorize = False


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
        if le_factorize:
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


def FE(df):
    
        df['day_x_age']=df['day']*df['age']
        df['day_divide_age']=df['day']/(df['age']+1)
    
        for c in ['balance','duration','campaign','previous']:
            for i in range(1,6):
                df[f'{c}_{i}']=df[c]//(10**(i))%10
            
        df['month']=df['month'].map({'aug':8, 'jun':6, 'may':5, 'feb':2, 'apr':4, 
            'nov':11, 'jul':7, 'jan':1, 'oct':10,'mar':3, 'sep':9, 'dec':12})
        #df['date']=df['month'].astype(str)+"_"+df['day'].astype(str)
        df['total_days']=df['month']*31+df['day']
    
        df['sin_duration']=np.sin(2*np.pi*df['duration']/365)
        df['cos_duration']=np.cos(2*np.pi*df['duration']/365)
        
        df['sin_age']=np.sin(2*np.pi*df['age']/10)
        df['cos_age']=np.cos(2*np.pi*df['age']/10)
    
        df['sin_balance']=np.sin(2*np.pi*df['balance']/75)
        df['cos_balance']=np.cos(2*np.pi*df['balance']/75)
    
        df['sin_day']=np.sin(2*np.pi*df['day']/10)
        df['cos_day']=np.cos(2*np.pi*df['day']/10)

        return df


def CatEncode(train_df, test_df, feats, target):
    # Simple Label encode
    if 1:
        for col in feats:
            le = LabelEncoder().fit(
                np.unique(train_df[col].unique().tolist()+
                          test_df[col].unique().tolist()))                    
            le_mapping = dict(zip(le.classes_, range(len(le.classes_))))    
            #At the end 0 will be used for null values so we start at 1 
            train_df[col] = le.transform(train_df[col])
            test_df[col]  = le.transform(test_df[col])
        
            if 'nan' in le_mapping:
                train_df[col] = train_df[col].replace(le_mapping['nan'], np.nan)
                test_df[col]  = test_df[col].replace(le_mapping['nan'], np.nan)
    # Weight of evidence
    if 0:
        feats = feats
        woe_encoder = WoEEncoder(variables=feats, ignore_format=True)
        woe_encoder.fit(train_df, train_df[target])
        train_df = woe_encoder.transform(train_df)
        test_df = woe_encoder.transform(test_df)
    # Ratio of the probability of the target
    if 0:
        feats = feats
        pre_encoder = PRatioEncoder(variables=feats, ignore_format=True)
        pre_encoder.fit(train_df, train_df[target])
        train_df = pre_encoder.transform(train_df)
        test_df = pre_encoder.transform(test_df)
    # Rare or infrequent categories (Rare category should be replaced after appling this step)
    if 0:
        feats = feats
        rare_encoder = RareLabelEncoder(variables=feats, ignore_format=True)
        rare_encoder.fit(train_df, train_df[target])
        train_df = rare_encoder.transform(train_df)
        test_df = rare_encoder.transform(test_df)

    return train_df, test_df


if 1:
    train_df = FE(train.copy())    
    test_df = FE(test.copy())

    CATS = []
    for c in train_df.columns:    
        if train_df[c].dtype=="object":
            CATS.append(c)

    train_df, test_df = CatEncode(train_df, test_df, CATS, target)
else:
    train_df = train.copy()
    test_df = test.copy()


sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=train_df[target]
)


xgb_clf = XGBClassifier(tree_method='hist', 
                                device=device,                                  
                                eval_metric='auc',
                                random_state=2025,
                                sample_weight=sample_weights,
                                eta=0.01,
                                n_estimators=10000
                                )


FEATURES = list(set(train_df)-set(target))


%%time

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2025)

oof = np.zeros(len(train_df))
pred = np.zeros(len(test_df))
roc_scores = []

for i, (train_index, test_index) in enumerate(kf.split(train_df[FEATURES], train_df[target])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_df.loc[train_index,FEATURES].copy()
    y_train = train_df.loc[train_index,"y"]
    x_valid = train_df.loc[test_index,FEATURES].copy()
    y_valid = train_df.loc[test_index,"y"]
    x_test = test_df[FEATURES].copy()


    model = clone(xgb_clf)
    model.fit(x_train, y_train, 
              eval_set=[(x_valid, y_valid)], 
              early_stopping_rounds=200, 
              verbose=1000
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




