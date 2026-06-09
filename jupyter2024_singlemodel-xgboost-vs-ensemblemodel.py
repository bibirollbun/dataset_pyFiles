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


# Import Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import spearmanr
from itertools import combinations
import optuna
from scipy.stats import chi2_contingency

import xgboost as xgb
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score,f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,StackingClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier,Pool
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from optuna.samplers import TPESampler
import logging
import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Import Dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
orginal = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
orginal_a = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


train.head()


# As we can see that the columns of id is useless,we need to remove it
train = train.drop('id',axis=1,errors='ignore')
test = test.drop('id',axis=1,errors='ignore')


# Now,We can check the basic information of data
print("Train_shape:",train.shape,"||","Test_shape:",test.shape)
print ('='*50)
print("Train_missing_ratio:\n",train.isna().mean(),'\n'+'-'*40,"\nTest_missing_ratio:\n",test.isna().mean())
print('='*50)
print(train.info())


# Now we split the features to different data type
Target='Personality'
cat_cols = test.select_dtypes('object').columns.tolist()
num_cols = train.select_dtypes('float64').columns.tolist()
concat_cols = test.columns.tolist()


# view the distribution of missing data 
train_features_miss = train.drop(Target,axis=1).isna().mean()
test_features_miss = test.isna().mean()

fig,ax = plt.subplots(figsize=(10,6))
train_features_miss.plot(ax=ax,label='Train',marker='o')
test_features_miss.plot(ax=ax,label='Test',marker='x')

ax.set_title('Train_miss_VS_Test_miss')
ax.set_xlabel('Features')
ax.set_ylabel('missing_ratio')
plt.xticks(rotation=45,ha='right')
ax.legend()
ax.grid(True)
plt.show()


# Now maybe we need to the features 
plt.figure(figsize=(15,20))
for i,col in enumerate(concat_cols):
    plt.subplot(4,2,i+1)
    sns.histplot(data=train,x=col,hue=Target,kde=True,edgecolor='red',multiple='dodge')
    plt.title(f'Distribution of {col}',fontsize=14)
    plt.xlabel(col,fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.grid(axis='y',linestyle='--',alpha=0.9)
plt.tight_layout()
plt.show()


# Now,we can see the num_features distribution
plt.figure(figsize=(14,6))
for i,col in enumerate(num_cols):
    plt.subplot(2,3,i+1)
    sns.boxplot(data=train,y=col,palette='Set2')
    plt.title(f'Boxplot:{col}')

plt.tight_layout()
plt.show()


# 3. We need to analysis the target_feature distribution
plt.figure(figsize=(10,6))
ax = sns.countplot(data=train,x=Target,palette='pastel',edgecolor='black')
ax.bar_label(ax.containers[0])
plt.title('distribution of Personality types',fontsize=14)
plt.xlabel('Personality Type',fontsize=12)
plt.grid(axis='y',linestyle='--',alpha=0.5)
plt.tight_layout()
plt.show()

print("\nğŸ“Š Personality Value Counts (Proportions):")
print(train[Target].value_counts(normalize=True).round(3))


#4.The relation of num_features to analysis
plt.figure(figsize=(6,4))
sns.heatmap(train[num_cols].corr(),annot=True,cmap='YlGnBu',fmt='.2f')
plt.title('Correlation Between Numerical Features')
plt.show()


# 5. num_features importance by Chi-square test
results = []
for feature in num_cols:
    # åˆ›å»ºåˆ—è�”è¡¨
    contingency_table = pd.crosstab(train[feature],train[Target])
    # æ‰§è¡Œå�¡æ–¹æ£€éªŒ
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results.append({
        'Feature': feature,
        'Chi2': chi2,
        'p-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })

# ç»“æ�œæ�’åº�
results_df = pd.DataFrame(results).sort_values('Chi2', ascending=False)
print(results_df)


# As we know,we have two orginal dataset,one has missing data but another not.we know they the same data,so we can find how to fill missing data by analysis them.
num_missing_cols = {}
for col in num_cols:
    a = (orginal[orginal_a[col].isna()][col] == orginal_a[col].mean())
    b = a.unique()
    num_missing_cols[col] = b

cat_missing_cols = {}
for col in cat_cols:
    orginal[orginal_a[col].isna()][col] == orginal_a[col].mode()[0]
    b = a.unique()
    cat_missing_cols[col] = b

print('num_cols is filled by average:\n',num_missing_cols)
print('='*100)
print('num_cols is filled by mode:\n',cat_missing_cols)




# Fill Missing data:how to fill the missing data ,num_features for average and cat_features for mode
for i in num_cols:
    train[i] = train[i].fillna(train[i].mean())
    test[i] = test[i].fillna(test[i].mean())
for i in cat_cols:
    train[i] = train[i].fillna(train[i].mode()[0])
    test[i] = test[i].fillna(test[i].mode()[0])


# cat_features Encoding
for i in cat_cols:
    train[i] = np.where(train[i] == 'No',0,1)
    test[i] = np.where(test[i] == 'No',0,1)

train[Target] = np.where(train[Target] == 'Introvert',0,1)


#.The relation of ALL_features to analysis
plt.figure(figsize=(6,4))
sns.heatmap(train[concat_cols].corr(),annot=True,cmap='YlGnBu',fmt='.2f')
plt.title('Correlation Between Numerical Features')
plt.show()


#  ALL_features importance by Chi-square test
results = []
for feature in concat_cols:
    # åˆ›å»ºåˆ—è�”è¡¨
    contingency_table = pd.crosstab(train[feature],train[Target])
    # æ‰§è¡Œå�¡æ–¹æ£€éªŒ
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results.append({
        'Feature': feature,
        'Chi2': chi2,
        'p-value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })

# ç»“æ�œæ�’åº�
results_df = pd.DataFrame(results).sort_values('Chi2', ascending=False)
print(results_df)


X = train.drop(Target,axis=1)
y = train[Target]

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# Dmatrix
dtrain = xgb.DMatrix(X_train,label=y_train)
dval = xgb.DMatrix(X_val,label=y_val)
dtest = xgb.DMatrix(test)


# xgboost  
params = {
    'objective': 'binary:logistic',  
    'eval_metric': ['logloss', 'auc', 'error'], 
    'eta': 0.05, 
    'max_depth': 4,  
    'min_child_weight': 2,  
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'gamma': 0,  
    'lambda': 1,  
    'alpha': 0,  
    'scale_pos_weight': 1,  
    'random_state': 42  
}
evals_result = {} 
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  
    evals=[(dtrain, 'train'), (dval, 'val')],  
    early_stopping_rounds=20, 
    evals_result=evals_result,  
    verbose_eval=10  
)



# Test_predict
y_pred_proba = model.predict(dtest)
y_pred = np.round(y_pred_proba)

y_pred_reverse = ['Extrovert' if x == 1 else 'Introvert' for x in y_pred]
submission['Personality'] = y_pred_reverse
submission.to_csv('/kaggle/working/submission.csv',index=False)


submission.head()


# Importance of Feature 
importance_dict = model.get_score(importance_type='gain') 
importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
}).sort_values('Importance', ascending=False)

print(importance_df)


n_folds = 5
skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)


# Model Params setting

xgb_params = {'learning_rate': 0.0977045436969459,
 'max_depth': 6,
 'min_child_weight': 3.4028275725424613,
 'subsample': 0.7817216529980646,
 'colsample_bytree': 0.7808046762745895,
 'gamma': 1.7773050649152045,
 'reg_alpha': 2.9119645865661113e-07,
 'reg_lambda': 0.09763010168885818,
 'scale_pos_weight': 2.7816962287054956,
 'n_estimators': 10000,
 'objective': 'binary:logistic',
 'random_state': 42}

cat_params = {'iterations': 1500,
              'learning_rate': 0.08883411654598307, 
              'depth': 5, 
              'l2_leaf_reg': 0.012565682664018613, 
              'random_strength': 0.6751209862830547,
              'bagging_temperature': 1.865641154955605,
              'grow_policy': 'Depthwise', 
              'min_data_in_leaf': 26, 
              'auto_class_weights': 'Balanced',
              'loss_function': 'Logloss',
              'verbose': False,
              'random_seed': 42}

lgb_params = {'learning_rate': 0.006645571568372025,
              'n_estimators': 1000, 
              'num_leaves': 24, 
              'max_depth': 5, 
              'min_child_samples': 38, 
              'subsample': 0.9362148766913249,
              'colsample_bytree': 0.7222387389288989, 
              'reg_alpha': 2.1947395275856394, 
              'reg_lambda': 0.8088332704222071, 
              'scale_pos_weight': 3.740332552613225,    
              'objective': 'binary',
              'metric': 'auc',
              'n_estimators': 1000,
              'random_state': 42}

# Create meta matrix
train_meta = np.zeros((X.shape[0],3))
test_meta = np.zeros((test.shape[0],3))
test_meta_folds = np.zeros((test.shape[0],3,n_folds))



# # 5_splits Training for 1st layers
# for fold,(train_idx,valid_idx) in enumerate(skf.split(X,y)):
#     print (f"Training Fold {fold+1}/{n_folds}")

#     # split traing_dataset and val_dataset
#     X_train,X_val = X.iloc[train_idx],X.iloc[valid_idx]
#     y_train,y_val = y.iloc[train_idx],y.iloc[valid_idx]

#     # XGBOOST
#     xgb = XGBClassifier(**xgb_params)
#     xgb.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
#     train_meta[valid_idx,0] = xgb.predict_proba(X_val)[:,1]
#     test_meta_folds[:,0,fold] = xgb.predict_proba(test)[:,1]

#     # CatBoost
#     cat = CatBoostClassifier(**cat_params)
#     cat.fit(X_train,y_train,eval_set=(X_val,y_val),verbose=False)
#     train_meta[valid_idx,1] = cat.predict_proba(X_val)[:,1]
#     test_meta_folds[:,1,fold] = cat.predict_proba(test)[:,1]

#     # LightGBM
#     lgb = LGBMClassifier(**lgb_params)
#     lgb.fit(X_train,y_train,eval_set=[(X_val,y_val)],eval_metric='auc')
#     train_meta[valid_idx,2] = lgb.predict_proba(X_val)[:,1]
#     test_meta_folds[:,2,fold] = lgb.predict_proba(test)[:,1]

# # avg_test
# test_meta = test_meta_folds.mean(axis=2)    


# # 2nd Logistic Train
# meta_model = LogisticRegression(
#     penalty='l2', 
#     C=0.1, 
#     solver='lbfgs', 
#     max_iter=1000,
#     class_weight='balanced'  
# )

# meta_model.fit(train_meta,y)

# # final_predict
# final_predictions = meta_model.predict_proba(test_meta)[:,1]


# y_pred = np.round(final_predictions)
# y_pred_reverse = ['Introvert' if x == 0 else 'Extrovert' for x in y_pred]
# submission['Personality'] = y_pred_reverse
# submission.to_csv('/kaggle/working/submission.csv',index=False)

