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


import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm
warnings.filterwarnings("ignore")


class config:
    base_url="/kaggle/input/mercor-cheating-detection"
    train_csv =os.path.join(base_url,"train.csv")
    test_csv = os.path.join(base_url,"test.csv")
    social_graph = os.path.join(base_url,"social_graph.csv")
    sample_submission = os.path.join(base_url,"sample_submission.csv")


sample_sub=pd.read_csv(config.sample_submission).head()
sample_sub


train_df=pd.read_csv(config.train_csv)
train_df


train_df.is_cheating.unique()



(train_df.isnull().sum()/train_df.shape[0]).plot(kind="bar",title="Null value in percentage%")
plt.show()



print(train_df.isnull().sum().sum() - train_df.shape[0])
train_df.shape


df = train_df.dropna(subset=["is_cheating"])
df.drop(columns=['high_conf_clean',"user_hash"],inplace=True)
df.shape


df.isna().sum()


from sklearn.impute import SimpleImputer
emputer=SimpleImputer()
# def fit_imputer(df:pd.DataFrame):
#     cols=[col for col in df.columns if df[col].isnull().any()]
#     for col in cols:
#         df[col]=emputer.fit_transform(df[[col]])
#     return df
df_impute=df


df_impute.is_cheating.value_counts().plot(kind="bar",title=f"{df_impute.is_cheating.value_counts()}")
plt.show()


counts=df_impute.is_cheating.value_counts()
counts.plot(kind="bar",title=f"1.0 value= {counts[1]} and \n0.0 value = 40000")
plt.axhline(y=40000,linestyle="--",color="red")
plt.show()


from sklearn.utils import resample
df_0=df_impute[df_impute.is_cheating==0.0]
df_1=df_impute[df_impute.is_cheating==1.0]
print("0.0 ->",len(df_0),'1.0 ->',len(df_1))
print("Balancing...")

df_0_down=resample(df_0,
                  replace=False,
                  n_samples=34431,
                  random_state=42)
print("Done balancing")
df_1_n_0=pd.concat([df_0_down,df_1])
df_balanced = df_1_n_0.sample(frac=1, random_state=42).reset_index(drop=True)


df_balanced.is_cheating.value_counts().plot(kind="bar",title="After balancing",color="green")
plt.show()


print(df_balanced.shape)
df_balanced.isnull().sum().any()


from sklearn.model_selection import train_test_split
x=df_balanced.drop(columns=["is_cheating"],axis=1)
y=df_balanced["is_cheating"]
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,
                                               random_state=42,
                                                stratify=y)
feature_names = x_train.columns
x_train=emputer.fit_transform(x_train)
x_test=emputer.transform(x_test)
x_train = pd.DataFrame(x_train, columns=feature_names)
x_test = pd.DataFrame(x_test, columns=feature_names)
print("x_train shape :",x_train.shape,"and  y_train shape :",y_train.shape)
print("x_test shape :",x_test.shape,"and  y_test shape :",y_test.shape)


y_train.value_counts().plot(kind="barh",color="yellow",title="checking data splitted equaly")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn import tree
from  xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import check_scoring
from sklearn.metrics import classification_report,accuracy_score


import optuna
lr=LogisticRegression(class_weight="balanced",C=10)
lro=lr.fit(x_train,y_train)
score_lr=check_scoring(lro,scoring="accuracy")
print("Test score LR =",score_lr(lro,x_test,y_test))
print("Train scoe LR=",score_lr(lro,x_train,y_train))
print("classification_report LR :\n")
print(classification_report(y_test,lr.predict(x_test)))
print("="*50)

catb=CatBoostClassifier(iterations= 1000, 
                        # depth= 8, 
                        # learning_rate=0.038807323069239986,
                        # l2_leaf_reg= 0.0010721277180962543, 
                        # bagging_temperature= 0.5038207452382065, 
                        # random_strength= 4.523816842643959
                       )
catb.fit(x_train,y_train)
scorecatb=check_scoring(catb,scoring="accuracy")
print("Test scorecatb =",scorecatb(catb,x_test,y_test))
print("Train scoecatb=",scorecatb(catb,x_train,y_train))
print("classification_reportcatb :\n")
print(classification_report(y_test,catb.predict(x_test)))
print("="*50)

tree=tree.DecisionTreeClassifier(class_weight="balanced")
tree.fit(x_train,y_train)
score_tree=check_scoring(tree,scoring="accuracy")
print("Test score tree =",score_tree(tree,x_test,y_test))
print("Train scoe tree=",score_tree(tree,x_train,y_train))
print("classification_report tree :\n")
print(classification_report(y_test,tree.predict(x_test)))

print("="*55)
rfc=RandomForestClassifier(class_weight="balanced",
                          min_samples_leaf=20)
rfc.fit(x_train,y_train)
score_rfc=check_scoring(rfc,scoring="accuracy")
print("Test score rfc =",score_rfc(rfc,x_test,y_test))
print("Train scoe rfc=",score_rfc(rfc,x_train,y_train))
print("classification_report rfc :\n")
print(classification_report(y_test,rfc.predict(x_test)))

print("="*55)
xgb=XGBClassifier(class_weight="balanced")
xgb.fit(x_train,y_train)
score_xgb=check_scoring(xgb,scoring="accuracy")
print("Test score xgb =",score_xgb(xgb,x_test,y_test))
print("Train scoe xgb=",score_xgb(xgb,x_train,y_train))
print("classification_report xgb :\n")
print(classification_report(y_test,xgb.predict(x_test)))



print("="*55)
light=LGBMClassifier(class_weight="balanced")
light.fit(x_train,y_train)
score_light=check_scoring(light,scoring="accuracy")
print("Test score light =",score_light(light,x_test,y_test))
print("Train scoe light=",score_light(light,x_train,y_train))
print("classification_report light :\n")
print(classification_report(y_test,light.predict(x_test)))

# print("Applying stacking method..")
# kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof_rf=np.zeros(len(x_train)) #out of fold(OOf)
# oof_xgb=np.zeros(len(x_train))
# oof_light=np.zeros(len(x_train))
# rf=RandomForestClassifier(n_estimators=500,
#                             class_weight="balanced",
#                             min_samples_leaf=20)
# xgbs=XGBClassifier(n_estimators=2999,
#                     scale_pos_weight=5)
# lgb=LGBMClassifier(class_weight="balanced")
# for train_idx,val_idx in kf.split(x_train,y_train):
#     x_tr,x_val=x_train.iloc[train_idx],x_train.iloc[val_idx]
#     y_tr,y_val = y_train.iloc[train_idx],y_train.iloc[val_idx]

#     rf.fit(x_tr,y_tr)
#     xgbs.fit(x_tr,y_tr)
#     lgb.fit(x_tr,y_tr)

#     oof_rf[val_idx] = rf.predict_proba(x_val)[:,1]
#     oof_xgb[val_idx] = xgbs.predict_proba(x_val)[:,1]
#     oof_light[val_idx] = lgb.predict_proba(x_val)[:,1]

# x_meta=np.column_stack([oof_rf,oof_xgb,oof_light])
# meta_model =LogisticRegression()
# meta_model.fit(x_meta,y_train)

# rf.fit(x_train, y_train)
# xgb.fit(x_train, y_train)
# lgb.fit(x_train, y_train)

# test_rf = rf.predict_proba(x_test)[:, 1]
# test_xgb = xgb.predict_proba(x_test)[:, 1]
# test_lgb = lgb.predict_proba(x_test)[:, 1]

# X_meta_test = np.column_stack([test_rf, test_xgb, test_lgb])
# meta_prob = meta_model.predict_proba(X_meta_test)[:, 1]
# print("Meta-Model ROC-AUC:", roc_auc_score(y_test, meta_prob))

# meta_pred = (meta_prob >= 0.5).astype(int)

# print("Classification Report (Meta Model):\n")
# print(classification_report(y_test, meta_pred))



y_prob = rfc.predict_proba(x_test)[:, 1]
  
roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score for rfc:", roc_auc)

print("-+"*50)
y_prob_lr=lr.predict_proba(x_test)[:, 1]
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)
print("ROC-AUC Score for lr:", roc_auc_lr)

print("-+"*50)
y_prob_xgb=xgb.predict_proba(x_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print("ROC-AUC Score for xgb:", roc_auc_xgb)

print("-+"*50)
y_prob_light=light.predict_proba(x_test)[:, 1]
roc_auc_light = roc_auc_score(y_test, y_prob_light)
print("ROC-AUC Score for light:", roc_auc_light)

print("-+"*50)
y_prob_catb=catb.predict_proba(x_test)[:, 1]
roc_auc_catb = roc_auc_score(y_test, y_prob_catb)
print("ROC-AUC Score for catb:", roc_auc_catb)# 0.8383345336427751 




social_graph=pd.read_csv(config.social_graph).head()
social_graph 


x_train.columns


test_df=pd.read_csv(config.test_csv)
test_df.head()


clone_test=test_df.copy()
clone_test.drop(columns=["user_hash"],inplace=True)



clone_test=emputer.transform(clone_test)
clone_test=pd.DataFrame(clone_test, columns=feature_names)
clone_test.isna().sum().any()


clone_test.head()


# test_rf = rf.predict_proba(clone_test)[:, 1]
# test_xgb = xgb.predict_proba(clone_test)[:, 1]
# test_lgb = lgb.predict_proba(clone_test)[:, 1]

# meta_test = np.column_stack([test_rf, test_xgb, test_lgb])
# meta_prob = meta_model.predict_proba(meta_test)[:, 1]
test_catb = catb.predict_proba(clone_test)[:, 1]


len(test_catb)==test_df.shape[0]


sub_mission=pd.DataFrame({
    "user_hash" :test_df.user_hash,
    "prediction" : test_catb 
})


sub_mission.head()


#sub_mission.to_csv("Submission2.csv",index=False)

