!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!cp /kaggle/input/metric/metric.py ./
from metric import score
import numpy as np, pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
import os
import pickle
import joblib
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
race_groups = train['race_group'].unique()
for race in race_groups:
    print(race)
    race_df = train[train['race_group'] == race]
    print(len(race_df[race_df['efs'] == 1])/len(race_df))


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
train=train[train["race_group"]=="White"]
test=test[test["race_group"]=="White"]
print(train.loc[train.efs==1,"efs_time"].describe())

plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
train=train[train["race_group"]=="Asian"]
test=test[test["race_group"]=="Asian"]
print(train.loc[train.efs==1,"efs_time"].describe())

plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
train.loc[train.efs==0,"y"] -= 0.15


plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
# plt.xlim((0,0.01))
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]

CATS = []
for c in FEATURES:
    num_unique = train[c].nunique()
    if num_unique < 100:
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")

combined = pd.concat([train,test],axis=0,ignore_index=True)

encoders = {}
for c in FEATURES:
    if c in CATS:
        encoders[c] = LabelEncoder()
        combined[c] = encoders[c].fit_transform(combined[c].astype(str))
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)


FOLDS = 10
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=1)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(skf.split(train, train.race_group)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=20000,  
        learning_rate=0.01,  
        enable_categorical=True,
        min_child_weight=80,
        early_stopping_rounds=500,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=1000 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    model_path = os.path.join('/kaggle/working/', f"xgb_fold_{i+1}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model_xgb, file)
        print(f"Model for Fold {i+1} saved at {model_path}")


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)

