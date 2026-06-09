import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
df_test_ov = df_test.copy()


df_orig=pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")
df_orig=df_orig.rename(columns={'RowNumber':'id'})


def getGrps(df_orig, train,test, grpCols):


    # # 建立 rounded 欄位，不修改原始欄位
    # train['Age_round'] = train['Age'].round(-1)
    # test['Age_round'] = test['Age'].round(-1)
    # df_orig['Age_round'] = df_orig['Age'].round(-1)

    # train['CreditScore_round'] = train['CreditScore'].round(-1)
    # test['CreditScore_round'] = test['CreditScore'].round(-1)
    # df_orig['CreditScore_round'] = df_orig['CreditScore'].round(-1)

    # train['EstimatedSalary_round'] = train['EstimatedSalary'].round(-4)
    # test['EstimatedSalary_round'] = test['EstimatedSalary'].round(-4)
    # df_orig['EstimatedSalary_round'] = df_orig['EstimatedSalary'].round(-4)

    # 開始針對各欄位 groupby
    grpBy=[]
    for c in grpCols:
        grpBy.append(c)
        # 計算 group mean
        df_tmp = df_orig.groupby(grpBy).agg({'id':'count','Exited':{'sum'}}).reset_index()
        new_col =list(map(''.join, df_tmp.columns.values))
        df_tmp.columns = [col if col in grpBy else f'{c}_{col}_org1' for col in new_col]
        #print(df_tmp.columns)
        # 合併進資料
        train = train.merge(df_tmp, how='left')
        test = test.merge(df_tmp, how='left')
        #print(f"[{c}] train NaN = {train.isnull().sum()}, test NaN = {test[new_col].isnull().sum()}")
        n_c=df_tmp.columns.drop(grpBy)
        train[n_c]= train[n_c].fillna(0).astype(int)
        test[n_c]= test[n_c].fillna(0).astype(int)

    return train, test
grpCols=['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'Tenure', 'CreditScore', 
         'NumOfProducts','HasCrCard', 
         'IsActiveMember' ,'EstimatedSalary','Balance']

df_train_org1, df_test_org1 = getGrps(df_orig, df_train, df_test, grpCols)

def getGrpsIndv(df_orig,df_train,df_test,grpCols):
    grpBy=[]
    # df_orig['Age_round'] = df_orig['Age'].round(-1)
    # df_orig['CreditScore_round'] = df_orig['CreditScore'].round(-1)
    # df_orig['EstimatedSalary_round'] = df_orig['EstimatedSalary'].round(-4)
    for c in grpCols:
        for i in grpCols:
            if c!=i:
                grpBy=[c,i]
                df_tmp=df_orig.groupby(grpBy).agg({'id':'count','Exited':{'sum'}}).reset_index()
                df_tmp.columns=list(map(''.join, (list(df_tmp.columns.values))))
                sepCols=df_tmp.columns.drop(grpBy)+'_Orig_groups_ind_'+c+'_'+i
                df_tmp.columns=list(grpBy)+list(sepCols)
                #
                df_train=df_train.merge(df_tmp,how='left')
                df_test=df_test.merge(df_tmp,how='left')
                df_train[sepCols]=df_train[sepCols].fillna(0)
                df_test[sepCols]=df_test[sepCols].fillna(0)

                df_train[sepCols]=df_train[sepCols].astype('int')
                df_test[sepCols]=df_test[sepCols].astype('int')
    return df_train,df_test

grpCols=['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'Tenure', 'CreditScore', 
         'NumOfProducts','HasCrCard', 
         'IsActiveMember' ,'EstimatedSalary','Balance']
df_train_org2,df_test_org2=getGrpsIndv(df_orig,df_train_org1,df_test_org1,grpCols)


def add_group_stats(df_train, df_test):
    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

    aggs = {
        'Age': ['min','max', 'mean'],       
        'Balance': ['min','max', 'mean','sum'],
        'NumOfProducts': ['mean','sum'],
        'IsActiveMember': ['min','max', 'mean','sum'],
        'CreditScore': ['min','max', 'mean'],
        'EstimatedSalary': ['min','max', 'mean','sum'],
        'id': 'count',
    }

    # Group 1: ['Surname', 'Geography', 'Gender']
    group_keys1 =['CustomerId', 'Surname', 'Geography', 'Gender']
    df_grp1 = df_all.groupby(group_keys1).agg(aggs).reset_index()
    new_columns=list(map(''.join, df_grp1.columns.values))
    df_grp1.columns=[col if col in group_keys1 else f'{col}_g1' for col in new_columns]
    # 攤平欄名並加 g1_ prefix
    
    print(df_grp1.columns)
    # Group 2: ['Geography', 'Gender']
    group_keys2 = ['CustomerId', 'Surname', 'Age', 'Gender']
    df_grp2 = df_all.groupby(group_keys2).agg(aggs).reset_index()
    new_columns=list(map(''.join, df_grp2.columns.values))
    df_grp2.columns=[col if col in group_keys2 else f'{col}_g2' for col in new_columns]
    print(df_grp2.columns)
    # 合併進 train/test
    df_train = df_train.merge(df_grp1, how='left', on=group_keys1)
    df_test = df_test.merge(df_grp1, how='left', on=group_keys1)
    df_train = df_train.merge(df_grp2, how='left', on=group_keys2)
    df_test = df_test.merge(df_grp2, how='left', on=group_keys2)

    return df_train, df_test



df_train_merge,df_test_merge=add_group_stats(df_train_org2,df_test_org2)


import numpy as np

def add_user_history_features(df_train, df_test):
    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

    exitGrpBy=['CustomerId', 'Surname',  'Gender','Geography','EstimatedSalary']

    exitSrtBy=['CustomerId', 'Surname',  'Gender','Geography','Age', 'Tenure']
    ##
    df_all_Exits=df_all.copy()
    df_all_Exits['Exited']=df_all_Exits['Exited'].fillna(-1)
    df_all_Exits=df_all_Exits.sort_values(exitSrtBy)
    df_all_Exits['Exit_lag1']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(1)
    df_all_Exits['Exit_lag2']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(2)
    df_all_Exits['Exit_lag3']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(3)
    
    df_all_Exits['Exit_lead1']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(-1)
    df_all_Exits['Exit_lead2']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(-2)
    df_all_Exits['Exit_lead3']=df_all_Exits.groupby(exitGrpBy)['Exited'].shift(-3)
    
    df_all_Exits['Balance_lag_diff1']=df_all_Exits['Balance'].shift(1)
    df_all_Exits['Balance_lead_diff1']=df_all_Exits['Balance'].shift(-1)
    #print(df_all_Exits.columns)
    df_all_Exits=df_all_Exits[['id','Exit_lag1','Exit_lag2','Exit_lag3',
                             'Exit_lead1','Exit_lead2','Exit_lead3',
                              'Balance_lag_diff1','Balance_lead_diff1']]
    df_all_Exits=df_all_Exits.fillna(-1).astype('int')
    df_train=df_train.merge(df_all_Exits,how='left')
    df_test=df_test.merge(df_all_Exits,how='left')
    return df_train, df_test
df_train_history,df_test_history= add_user_history_features(df_train_merge,df_test_merge)


def getFeats(df):
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']
    df['AgeCat'] = np.round(df.Age/20).astype('int').astype('category')
    df['Sur_Geo_Gend_Sal'] = df['Surname']+df['Geography']+df['Gender']+np.round(df.EstimatedSalary).astype('str')
    return df
df_train_feat = getFeats(df_train_history)
df_test_feat= getFeats(df_test_history)


feat_cols=df_train_feat.columns.drop(['id','Exited'])

print("Number of Features:",len(feat_cols))
print(feat_cols)
df_train_feat.head()


X=df_train_feat[feat_cols]
y=df_train_feat['Exited']
##
cat_features = np.where(X.dtypes != np.float64)[0]



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import category_encoders as ce
from sklearn.model_selection import train_test_split

# # 類別欄位
# obj_cols = df_train_org2.select_dtypes(include=['object','category']).columns.tolist()

# # 1. 拆分 train/test（可加 stratify）
# X = df_train_org2.drop(columns=['Exited'])
# y = df_train_org2['Exited']

# # 2. Target Encoding：fit on train only（避免洩漏）
# encoder = ce.TargetEncoder(cols=obj_cols)
# X_train_encoded = encoder.fit_transform(X, y)
#         # 注意：val 只 transform，不 fit
# df_test_encoded = encoder.transform(df_test_org2)      # 同上

# 3. 確保欄位一致
#print(X_train_encoded.columns == df_test_encoded.columns)





# import optuna
# # 交叉驗證

# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# def objective_xgb(trial):
#     # 定義參數搜尋空間
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 150),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'eval_metric': 'logloss',
#         'use_label_encoder': False,
#         'random_state': 42
#     }

#     model = XGBClassifier(**params)

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []

#     for train_idx, val_idx in kf.split(X_train_encoded, y):
#         X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
#             early_stopping_rounds=10,
#             verbose=False
#         )

#         proba = model.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, proba))

#     return np.mean(aucs)

# study_xgb = optuna.create_study(direction='maximize')
# study_xgb.optimize(objective_xgb, n_trials=100, timeout=2000)

# print("Best AUC:", study_xgb.best_value)
# print("Best Parameters:", study_xgb.best_params)


# # 建立模型們
# model_xgb =XGBClassifier(** study_xgb.best_params)
# auc_scores = []
# n_repeats = 5
# n_folds = 5

# test_preds = np.empty((n_repeats * n_folds, len(df_test_encoded)))

# for rep in range(n_repeats):
#     randomstate = 42 + rep
#     kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=randomstate)
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y)):
#         X_fold, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         # voting_clf_tune.fit(X_fold, y_fold)
        
#         # val_pred_proba = voting_clf_tune.predict_proba(X_val)[:, 1]

#         model_xgb.fit(X_fold, y_fold)
        
#         val_pred_proba = model_xgb.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, val_pred_proba)
#         auc_scores.append(auc)
#         print(f"[Repeat {rep+1} Fold {fold+1}] AUC: {auc:.4f}")
#         # 存機率預測
#         test_pred_proba = model_xgb.predict_proba(df_test_encoded)[:, 1]
#         test_preds[rep * n_folds + fold, :] = test_pred_proba

# # 統計
# print(f"\nAverage AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

# final_test_preds = test_preds.mean(axis=0)



RAND_VAL=42
num_folds=5 ## Number of folds
n_est=6000 ## Number of estimators


folds = StratifiedKFold(n_splits=num_folds,random_state=RAND_VAL,shuffle=True)
test_preds = np.empty((num_folds, len(df_test_feat)))
auc_vals=[]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    train_pool = Pool(X_train, y_train,cat_features=cat_features)
    val_pool = Pool(X_val, y_val,cat_features=cat_features)
    
    clf = CatBoostClassifier(
    eval_metric='AUC',
    task_type='GPU',
    learning_rate=0.02,
    iterations=6000)
    clf.fit(train_pool, eval_set=val_pool,verbose=300)
    
    y_pred_val = clf.predict_proba(X_val[feat_cols])[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print("AUC for fold ",n_fold,": ",auc_val)
    auc_vals.append(auc_val)
    
    y_pred_test = clf.predict_proba(df_test_feat[feat_cols])[:,1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")







join_cols=list(df_orig.columns.drop(['Exited']))
df_orig.rename(columns={'Exited':'Exited_Orig'},inplace=True)
df_orig['Exited_Orig']=df_orig['Exited_Orig'].map({0:1,1:0})
df_test_ov=df_test_ov.merge(df_orig,on=join_cols,how='left')[['id','Exited_Orig']].fillna(-1)
df_test_ov.head()


df_sub = df_test_ov[['id','Exited_Orig']]
per=(df_sub.Exited_Orig == -1).sum()/df_sub.shape[0]
print(f'percentage of y_pred:{per*100}')


y_pred = test_preds.mean(axis=0)

df_sub['Exited'] = np.where(df_sub.Exited_Orig==-1,y_pred,df_sub.Exited_Orig)
#df_sub['Exited']=y_pred
df_sub.drop('Exited_Orig',axis=1,inplace=True)
df_sub.head()


df_sub.to_csv("submission_1.csv",index=False)


df_sub_1=pd.read_csv('/kaggle/input/bankchurn-data-set/submission.csv')
df_sub_2=pd.read_csv('submission_1.csv')#original our output data
df_sub_3=pd.read_csv('/kaggle/input/bank-churn-2/submission_2.csv')
df_sub_4=pd.read_csv('/kaggle/input/bank-churn-3/submission_3.csv')

plt.figure(figsize=(12, 6))
plt.hist(df_sub_1['Exited'], bins=500, range=[0, 1], alpha=0.5, label='Model 1', color='blue')
plt.hist(df_sub_2['Exited'], bins=500, range=[0, 1], alpha=0.5, label='Model 2', color='green')
plt.hist(df_sub_3['Exited'], bins=500, range=[0, 1], alpha=0.5, label='Model 3', color='red')
plt.hist(df_sub_4['Exited'], bins=500, range=[0, 1], alpha=0.5, label='Model 3', color='yellow')

# 加上標籤與圖例
plt.xlabel('Exited')
plt.ylabel('Count')
plt.title('Histogram of Exited Values (Three Models)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df_ensemble = df_sub_1.copy()
df_ensemble['Exited'] = (
    df_sub_1['Exited'] + df_sub_2['Exited'] + df_sub_3['Exited']+df_sub_4['Exited']
) / 4

# 輸出檔案（如果需要）
df_ensemble.to_csv('submission.csv', index=False)




