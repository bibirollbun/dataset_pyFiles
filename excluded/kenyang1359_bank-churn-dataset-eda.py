import pandas as pd


df_train=pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")



print(f'train nan value={df_train.isnull().sum().sum()}')
print(f'test nan value={df_test.isnull().sum().sum()}')


df_orig=pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")
df_orig=df_orig.rename(columns={'RowNumber':'id'})


# æ‰¾å‡ºæ•¸å€¼å�‹æ¬„ä½�å’Œé¡�åˆ¥å�‹æ¬„ä½�
num_cols = df_orig.select_dtypes(include=['int','float']).columns
obj_cols = df_orig.select_dtypes(include=['object', 'string', 'category']).columns

# æ•¸å€¼æ¬„ä½�è£œ 0
df_orig[num_cols] = df_orig[num_cols].fillna(0)

# é¡�åˆ¥æ¬„ä½�è£œ 'na'
df_orig[obj_cols] = df_orig[obj_cols].fillna('na')


df_train.shape[0]


diff_custid = set(df_train['CustomerId']) - set(df_orig['CustomerId'])
diff_surname = set(df_train['Surname']) - set(df_orig['Surname'])
diff_custid_1 = set(df_orig['CustomerId']) - set(df_train['CustomerId'])
diff_surname_1 = set(df_orig['Surname']) - set(df_train['Surname'])

total_custid=len(diff_custid)+len(diff_custid_1)
total_surname=len(diff_surname)+len(diff_surname_1)
print("CustomerId diff of df_org df_train:", total_custid)
print("Surname diff of df_org df_train:", total_surname)


import matplotlib.pyplot as plt

# å·®é›†çµ±è¨ˆ
custid_only_in_test = len(set(df_test['CustomerId']) - set(df_orig['CustomerId']))
custid_only_in_orig = len(set(df_orig['CustomerId']) - set(df_test['CustomerId']))
custid_overlap = len(set(df_test['CustomerId']) & set(df_orig['CustomerId']))

surname_only_in_test = len(set(df_test['Surname']) - set(df_orig['Surname']))
surname_only_in_orig = len(set(df_orig['Surname']) - set(df_test['Surname']))
surname_overlap = len(set(df_test['Surname']) & set(df_orig['Surname']))

# ç•«åœ–
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("CustomerId and Surname Overlap Between Datasets", fontsize=14)

# CustomerId pie chart
axes[0].pie(
    [custid_overlap, custid_only_in_orig, custid_only_in_test],
    labels=['Overlap', 'Only in Original', 'Only in Test'],
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999', '#99ff99'],
    startangle=90
)
axes[0].set_title("CustomerId Distribution")

# Surname pie chart
axes[1].pie(
    [surname_overlap, surname_only_in_orig],
    labels=['Overlap', 'Only in Original'],
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999', '#99ff99'],
    startangle=90
)
axes[1].set_title("Surname Distribution")


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



for col in ['Surname','CustomerId','Geography', 'Gender','Age']:
    per=df_all[col].nunique()/len(df_all[col])
    print(f'{col} unique same percentage ={per*100}')



group_count_percentage = df_all.groupby(['Surname']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")




group_count_percentage = df_all.groupby(['CustomerId']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")



group_count_percentage = df_all.groupby(['Surname','CustomerId']).ngroups/len(df_all)
print(f"surname percent unique groups: {group_count_percentage*100}%")




group_count_percentage = df_all.groupby(['Surname', 'Gender', 'Geography']).ngroups/len(df_all)
print(f"percent unique groups: {group_count_percentage*100}%")



group_count_percentage = df_all.groupby(['CustomerId', 'Gender', 'Geography']).ngroups/len(df_all)
print(f"percent unique groups: {group_count_percentage*100}%")


group_count_percentage = df_all.groupby(['CustomerId','Surname', 'Gender', 'Geography']).ngroups/len(df_all)
print(f"percent unique groups: {group_count_percentage*100}%")


df_train.head()



from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


obj_feats=list(df_train.select_dtypes(include=['object','category']).columns)
df_x=df_train.drop('Exited',axis=1,inplace=False)
df_y=df_train['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=1
)
cat_interp = CatBoostClassifier(verbose=False, 
                                cat_features=obj_feats, 
                                early_stopping_rounds=200)

cat_interp.fit(X_train, y_train, eval_set=(X_test, y_test))

feature_importance = cat_interp.get_feature_importance()
feature_names = X_train.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)





import matplotlib.pyplot as plt


# ç¹ªåœ–
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])  # å��å�‘ç•«åœ–è®“æœ€å¤§åœ¨ä¸Š
plt.xlabel('Importance')
plt.title(f'Feature Importances')
plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, roc_auc_score

# å�‡è¨­ä½ æœ‰è¨“ç·´å®Œçš„ catboost_model å’Œ xgboost_model
cat_pred = cat_interp.predict(X_test)
print("CatBoost AUC:", roc_auc_score(y_test, cat_pred))


import warnings

# æŠ‘åˆ¶æœªä¾†è­¦å‘Š
warnings.simplefilter(action='ignore', category=FutureWarning)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# æ¬„ä½�æŒ‘é�¸
col_name = df_train.select_dtypes(include=['int64','float64']).columns
col_name = col_name.drop('Exited')

# è¨­å®šå­�åœ–åˆ—æ•¸èˆ‡è¡Œæ•¸
col = 5
row = int(np.ceil(len(col_name) / col))

# å»ºç«‹å­�åœ–
fig, axes = plt.subplots(row, col, figsize=(20, 10))  # å¯¬é«˜å›ºå®šï¼Œåœ–æ›´ç¾�è§€
axes = axes.flatten()
exit_palette = {0: '#1f77b4', 1: '#ff7f0e'}
# ç•«æ¯�å¼µå­�åœ–
for i, c in enumerate(col_name):
    uniquevalue = df_train[c].unique()
    
    if len(uniquevalue) > 20:
        sns.histplot(x=c, data=df_train, hue='Exited', kde=True, alpha=0.5, ax=axes[i],palette=exit_palette)
        axes[i].set_xlabel(c)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{c} vs Exited')
    else:
        exit_rates = df_train.groupby(c)['Exited'].mean() * 100
        sns.barplot(x=exit_rates.index, y=exit_rates, ax=axes[i],color='#1f77b4')
        axes[i].set_title(f'Avg Exit Rate by {c}')
        axes[i].set_ylabel('Exit Rate (%)')

# æ¸…é™¤å¤šé¤˜å­�åœ–
for j in range(len(col_name), len(axes)):
    fig.delaxes(axes[j])

# è¨­å®šç¸½æ¨™é¡Œ
fig.suptitle('Feature Distribution by Exited Status', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # ç‚ºç¸½æ¨™é¡Œç•™ç©ºé–“
plt.show()






# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
# from sklearn.metrics import classification_report, roc_auc_score

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# import category_encoders as ce
# from sklearn.model_selection import train_test_split

# # é¡�åˆ¥æ¬„ä½�
# obj_cols = df_train_history.select_dtypes(include='object').columns.tolist()

# # 1. æ‹†åˆ† train/testï¼ˆå�¯åŠ  stratifyï¼‰
# X = df_train_history.drop(columns=['Exited'])
# y = df_train_history['Exited']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # 2. Target Encodingï¼šfit on train onlyï¼ˆé�¿å…�æ´©æ¼�ï¼‰
# encoder = ce.TargetEncoder(cols=obj_cols)
# X_train_encoded = encoder.fit_transform(X_train, y_train)
# X_val_encoded = encoder.transform(X_val)          # æ³¨æ„�ï¼šval å�ª transformï¼Œä¸� fit
# df_test_encoded = encoder.transform(df_test_history)      # å�Œä¸Š

# # 3. ç¢ºä¿�æ¬„ä½�ä¸€è‡´
# #print(X_train_encoded.columns == df_test_encoded.columns)


# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt

# # 1. å€‹åˆ¥è¨“ç·´ä¸‰å€‹æ¨¡å�‹
# model_xgb.fit(X_train_encoded, y_train)
# model_lgb.fit(X_train_encoded, y_train)
# model_gbdt.fit(X_train_encoded, y_train)

# # 2. åˆ†åˆ¥è¨ˆç®— AUC
# auc_xgb = roc_auc_score(y_val, model_xgb.predict_proba(X_val_encoded)[:, 1])
# auc_lgb = roc_auc_score(y_val, model_lgb.predict_proba(X_val_encoded)[:, 1])
# auc_gbdt = roc_auc_score(y_val, model_gbdt.predict_proba(X_val_encoded)[:, 1])

# # 3. è¨­å®š AUC ä½œç‚ºæ¬Šé‡�
# weights = [auc_xgb, auc_lgb, auc_gbdt]
# model_names = ['XGBoost', 'LightGBM', 'GBDT']

# # 4. ç•«åœ“é¤…åœ–
# plt.figure(figsize=(6, 6))
# plt.pie(weights, labels=model_names, autopct='%1.1f%%', startangle=90)
# plt.title("VotingClassifier Model Weights by AUC")
# plt.show()

# # 5. ç”¨é€™å€‹æ¬Šé‡�é‡�æ–°çµ„ VotingClassifier
# voting_clf_weighted = VotingClassifier(
#     estimators=[
#         ('xgb', model_xgb),
#         ('lgb', model_lgb),
#         ('gbdt', model_gbdt)
#     ],
#     voting='soft',
#     weights=weights
# )

# voting_clf_weighted.fit(X_train_encoded, y_train)
# y_proba = voting_clf_weighted.predict_proba(X_val_encoded)[:, 1]
# print("Weighted ROC AUC:", roc_auc_score(y_val, y_proba))





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
# from sklearn.metrics import roc_auc_score
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier



# # å»ºç«‹æ¨¡å�‹å€‘
# model_xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
# model_lgb = LGBMClassifier(random_state=42)
# model_gbdt = GradientBoostingClassifier(random_state=42)

# voting_clf = VotingClassifier(
#     estimators=[
#         ('xgb', model_xgb),
#         ('lgb', model_lgb),
#         ('gbdt', model_gbdt)
#     ],
#     voting='soft',
#     weights=weights
# )




# # äº¤å�‰é©—è­‰
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# auc_scores = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train), 1):
#     X, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#     y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     voting_clf.fit(X, y)
#     val_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, val_pred_proba)
#     auc_scores.append(auc)
#     test_preds += voting_clf.predict_proba(df_test_encoded)[:, 1]

#     print(f"Fold {fold} AUC: {auc:.4f}")
# test_preds /= kf.get_n_splits()
# # çµ�æ�œ
# print(f"\nAverage AUC: {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
# from sklearn.metrics import roc_auc_score
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

# # # åˆ�å§‹åŒ– test é �æ¸¬ç¸½å’Œ
# test_preds = np.zeros(len(df_test_encoded))
# auc_scores = []

# # é‡�è¤‡çš„ random_state æ•¸ï¼ˆä¾‹å¦‚ï¼š5æ¬¡ä¸�å�Œ random seedï¼‰
# n_repeats = 3
# n_folds = 5  # æ¯�æ¬¡å¹¾æŠ˜äº¤å�‰é©—è­‰
# voting_clf = VotingClassifier(
#     estimators=[
#         ('xgb', model_xgb),
#         ('lgb', model_lgb),
#         ('gbdt', model_gbdt)
#     ],
#     voting='soft'
# )

# for seed in range(n_repeats):
#     seed=seed+42
#     print(f"\nğŸ”� Random State: {seed}")
#     kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train), 1):
#         # æ¨¡å�‹æ¯�è¼ªéƒ½é‡�æ–°å»ºï¼ˆé�¿å…�é‡�ç”¨èˆŠç‹€æ…‹ï¼‰
#         model_xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=seed)
#         model_lgb = LGBMClassifier(random_state=seed,verbose=0)
#         model_gbdt = GradientBoostingClassifier(random_state=seed)

 
#         X, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         voting_clf.fit(X, y)

#         # è©•ä¼°ç•¶å‰� fold AUC
#         val_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
#         auc = roc_auc_score(y_val, val_pred_proba)
#         auc_scores.append(auc)

#         # å°� test å�šé �æ¸¬ä¸¦åŠ ç¸½
#         test_preds += voting_clf.predict_proba(df_test_encoded)[:, 1]

#         print(f"  Fold {fold} AUC: {auc:.4f}")

# # å°� test å�šå¹³å�‡ï¼ˆå…± n_repeats * n_folds æ¬¡ï¼‰
# test_preds /= (n_repeats * n_folds)

# # # æœ€çµ‚è¼¸å‡º AUC å¹³å�‡èˆ‡æ¨™æº–å·®
# print(f"\nâœ… Final Average AUC: {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")



# import optuna

# def objective_xgb(trial):
#     # å®šç¾©å�ƒæ•¸æ�œå°‹ç©ºé–“
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
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

#     for train_idx, val_idx in kf.split(X_train_encoded,y_train):
#         X_tr, X_val = X_train_encoded.iloc[train_idx],X_train_encoded.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model.fit(X_tr, y_tr)
#         proba = model.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, proba))
        
#     return np.mean(aucs)


# study_xgb = optuna.create_study(direction='maximize')
# study_xgb.optimize(objective_xgb, n_trials=50)

# print("Best AUC:", study_xgb.best_value)
# print("Best Parameters:", study_xgb.best_params)


# model_xgb =XGBClassifier(**study_xgb.best_params)
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# aucs = []
# test_preds = np.zeros(len(df_test_encoded))
# for train_idx, val_idx in kf.split(X_train_encoded, y_train):
#     X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     model_xgb .fit(X_tr, y_tr)
#     proba = model_xgb.predict_proba(X_val)[:, 1]
#     aucs.append(roc_auc_score(y_val, proba))

#     test_preds +=model_xgb.predict_proba(df_test_encoded)[:, 1]


# df_orig.info()


# from sklearn.model_selection import StratifiedKFold
# from catboost import CatBoostClassifier, Pool
# num_folds=5 ## Number of folds
# folds = StratifiedKFold(n_splits=num_folds,random_state=42,shuffle=True)
# test_preds = np.empty((num_folds, len(df_test)))
# auc_vals=[]
# X = df_train_org2.drop(columns=['Exited'])
# y = df_train_org2['Exited']

# cat_features=list(df_train_org2.select_dtypes(include=['object','category']).columns)
# for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#     X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
#     train_pool = Pool(X_train, y_train,cat_features=cat_features)
#     val_pool = Pool(X_val, y_val,cat_features=cat_features)
#     clf = CatBoostClassifier(
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.02,
#         iterations=6000 )
#     clf.fit(train_pool, eval_set=val_pool,verbose=300)
    
#     y_pred_val = clf.predict_proba(X_val)[:,1]
#     auc_val = roc_auc_score(y_val, y_pred_val)
#     print("AUC for fold ",n_fold,": ",auc_val)
#     auc_vals.append(auc_val)
    
#     y_pred_test = clf.predict_proba(df_test_org2)[:,1]
#     test_preds[n_fold, :] = y_pred_test
#     print("----------------")


# df_test_ov = df_test.copy()
# join_cols=list(df_orig.columns.drop(['Exited']))
# df_orig.rename(columns={'Exited':'Exited_Orig'},inplace=True)
# df_orig['Exited_Orig']=df_orig['Exited_Orig'].map({0:1,1:0})

# df_test_ov=df_test_ov.merge(df_orig,on=join_cols,how='left')[['id','Exited_Orig']].fillna(-1)
# df_test_ov.head()


# y_pred = test_preds.mean(axis=0)
# # df_sub = df_test_ov[['id','Exited_Orig']]
# # df_sub['Exited'] = np.where(df_sub.Exited_Orig==-1,y_pred,df_sub.Exited_Orig)
# df_sub=pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
# df_sub['Exited'] = y_pred 
# # df_sub.drop('Exited_Orig',axis=1,inplace=True)
# df_sub.head()


# # preds=test_preds.mean(axis=0)
# # submision=pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
# # print((submision['id'] ==  df_test_encoded['id']).all())
# # submision['Exited']=preds
# # #submission.merge(submission_pred,on=['id'], how='left')
# df_sub.to_csv("submission.csv", index=False)                                



# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# auc_scores = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train), 1):
#     X, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#     y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     voting_clf_tune.fit(X, y)
#     val_pred_proba = voting_clf_tune.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, val_pred_proba)
#     auc_scores.append(auc)
    
#     print(f"Fold {fold} AUC: {auc:.4f}")

# # çµ�æ�œ
# print(f"\nAverage AUC: {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")
# # æ¨¡å�‹é �æ¸¬ï¼ˆå�‡è¨­ä½ ä½¿ç”¨çš„æ˜¯ voting_clfï¼‰
# y_pred = voting_clf_tune.predict(df_test_encoded)

# # å�–å‡ºå°�æ‡‰ idï¼ˆdf_test_encoded çš„ idï¼‰
# submission = pd.DataFrame({
#     "id": df_test_encoded["id"].values,
#     "Depression": y_pred
# })

# submission.to_csv("submission.csv", index=False)


# import optuna

# def objective_xgb(trial):
#     # å®šç¾©å�ƒæ•¸æ�œå°‹ç©ºé–“
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
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

#     for train_idx, val_idx in kf.split(X_train_encoded, y_train):
#         X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model.fit(X_tr, y_tr)
#         proba = model.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, proba))

#     return np.mean(aucs)

# def objective_lgb(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'random_state': 42
#     }

#     model = LGBMClassifier(**params)

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []

#     for train_idx, val_idx in kf.split(X_train_encoded, y_train):
#         X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model.fit(X_tr, y_tr)
#         proba = model.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, proba))

#     return np.mean(aucs)

# def objective_gbdt(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'random_state': 42
#     }

#     model = GradientBoostingClassifier(**params)

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []

#     for train_idx, val_idx in kf.split(X_train_encoded, y_train):
#         X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model.fit(X_tr, y_tr)
#         proba = model.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, proba))

#     return np.mean(aucs)





# study_xgb = optuna.create_study(direction='maximize')
# study_xgb.optimize(objective_xgb, n_trials=20)

# print("Best AUC:", study_xgb.best_value)
# print("Best Parameters:", study_xgb.best_params)

# # LightGBM
# study_lgb = optuna.create_study(direction='maximize')
# study_lgb.optimize(objective_lgb, n_trials=20)
# print("Best LGB AUC:", study_lgb.best_value)
# print("Best LGB Params:", study_lgb.best_params)

# # GBDT
# study_gbdt = optuna.create_study(direction='maximize')
# study_gbdt.optimize(objective_gbdt, n_trials=20)
# print("Best GBDT AUC:", study_gbdt.best_value)
# print("Best GBDT Params:", study_gbdt.best_params)



# # å»ºç«‹æ¨¡å�‹å€‘
# model_xgb =XGBClassifier(** study_xgb.best_params)
# model_lgb =  LGBMClassifier(**study_lgb.best_params)
# model_gbdt = GradientBoostingClassifier(**study_gbdt.best_params)

# voting_clf_tune = VotingClassifier(
#     estimators=[
#         ('xgb', model_xgb),
#         ('lgb', model_lgb),
#         ('gbdt', model_gbdt)
#     ],
#     voting='soft',
#     weights=weights
# )




# # äº¤å�‰é©—è­‰
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# auc_scores = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train), 1):
#     X, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
#     y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     voting_clf_tune.fit(X, y)
#     val_pred_proba = voting_clf_tune.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, val_pred_proba)
#     auc_scores.append(auc)
    
#     print(f"Fold {fold} AUC: {auc:.4f}")

# # çµ�æ�œ
# print(f"\nAverage AUC: {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")
# # æ¨¡å�‹é �æ¸¬ï¼ˆå�‡è¨­ä½ ä½¿ç”¨çš„æ˜¯ voting_clfï¼‰
# y_pred = voting_clf_tune.predict(df_test_encoded)

# # å�–å‡ºå°�æ‡‰ idï¼ˆdf_test_encoded çš„ idï¼‰
# submission = pd.DataFrame({
#     "id": df_test_encoded["id"].values,
#     "Depression": y_pred
# })

# submission.to_csv("submission.csv", index=False)

