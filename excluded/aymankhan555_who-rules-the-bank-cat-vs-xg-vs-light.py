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


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV , train_test_split ,RandomizedSearchCV

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,roc_auc_score,f1_score
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder ,OrdinalEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



df= pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



print(df.columns)
print(df.info())
df.head()


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()



print(df['y'].dtype)
print(df['y'].value_counts())
sns.countplot(data=df,x='y')



num_cols = df.select_dtypes(exclude=['object']).drop(['id','y'],axis=1).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

plt.figure(figsize=(15,10))
for i,col in enumerate(num_cols,1):
    plt.subplot(3,3,i)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


X = df[num_cols+cat_cols]
y = df.y

X_train ,X_valid , y_train, y_valid = train_test_split(X,y, test_size = 0.2 ,random_state=42)
X_train_cat = X_train #to put raw categorical values in Catbosst
X_valid_cat = X_valid

ordinal_encoder = OrdinalEncoder()
X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = ordinal_encoder.transform(X_valid[cat_cols])

#label encoder can also be used
# label_encoder = LabelEncoder()
# for col in cat_cols:
#     X_train[col] = label_encoder.fit_transform(X_train[col])
#     X_valid[col] = label_encoder.transform(X_valid[col])
    




# import optuna
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score

# def objective(trial):
#     param = {
#         'tree_method': 'gpu_hist',
#         'gpu_id': 0,
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#         'n_estimators': trial.suggest_int('estimators',200,1000),
#         'random_state': 42,
#         'verbosity': 0
#     }

#     model = XGBClassifier(**param)
#     model.fit(X_train, y_train,
#               eval_set=[(X_valid, y_valid)],
#               early_stopping_rounds=50,
#               verbose=False)
#     y_pred_proba = model.predict_proba(X_valid)[:, 1]
#     return roc_auc_score(y_valid, y_pred_proba)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=40)

# print("Best XGBoost params:", study.best_params)



#xgboost parameters are taken from optuna

xgb_model = XGBClassifier(
    learning_rate=0.050159730559911,
    max_depth=14,
    min_child_weight=7,
    subsample=0.9765849412638382,
    colsample_bytree=0.6212668125096665,
    reg_alpha=7.099866009100517,
    reg_lambda=5.085636550686968,
    n_estimators=930,
    random_state=42
)
xgb_model.fit(X_train,y_train)
pred = xgb_model.predict(X_valid)
proba = xgb_model.predict_proba(X_valid)[:, 1]

xgb_f1 = f1_score(y_valid,pred)
xgb_auc = roc_auc_score(y_valid,proba)


print("ğŸ”� ROC AUC:", xgb_auc)
print("ğŸ�¯ F1 Score:", xgb_f1)



# import optuna
# from lightgbm import LGBMClassifier
# from lightgbm import early_stopping

# from sklearn.metrics import roc_auc_score

# def objective(trial):
#     param= {
#         'device' :'gpu',
#         'boosting_type':'gbdt',
#         'learning_rate' : trial.suggest_float('learning_rate',0.01,0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#         'n_estimators': trial.suggest_int('n_estimators',100,1000),  # int here!
#         'random_state': 42,
#         'verbosity': -1
#     }
#     model = LGBMClassifier(**param)
#     model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_metric='auc',
#     callbacks=[early_stopping(stopping_rounds=50)]
# )
#     y_pred_proba = model.predict_proba(X_valid)[:,1]  # missing in your code
#     return roc_auc_score(y_valid, y_pred_proba)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=40)

# print("Best LightGBM params:", study.best_params)



#LightGBM
lgb_model = LGBMClassifier(
    learning_rate=0.07459282861701985,
    num_leaves=139,
    max_depth=14,
    min_child_samples=93,
    subsample=0.7337142954327327,
    colsample_bytree=0.781462628945671,
    reg_alpha=2.603318780335231,
    reg_lambda=4.254119374067986,
    n_estimators=778,
    random_state=42,
    verbosity=-1,
    device='gpu'
)
lgb_model.fit(X_train,y_train)
pred = lgb_model.predict(X_valid)
proba = lgb_model.predict_proba(X_valid)[:,1]


lgb_f1 = f1_score(y_valid,pred)
lgb_auc = roc_auc_score(y_valid,proba)


print("ğŸ”� ROC AUC:", lgb_auc)
print("ğŸ�¯ F1 Score:", lgb_f1)


# import optuna
# from catboost import CatBoostClassifier
# from sklearn.metrics import roc_auc_score

# def objective(trial):
#     params = {
#         'iterations': 1000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#         'loss_function': 'Logloss',
#         'eval_metric': 'AUC',
#         'random_seed': 42,
#         'verbose': 0,
#         'early_stopping_rounds': 50,
#         'task_type': 'GPU',       
#         'devices': '0'            
#     }
#     model = CatBoostClassifier(**params)
#     model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
#     proba = model.predict_proba(X_valid)[:, 1]
#     return roc_auc_score(y_valid, proba)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)

# print("Best parameters:", study.best_params)



#CATBOOST parameters are take from optuna 
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_valid_cat[col] = X_valid_cat[col].astype(str)

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.07320361687148269,
    depth=9,
    l2_leaf_reg=7.336500873337641,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    task_type="GPU" 
)
cat_model.fit(X_train_cat,y_train,cat_features=cat_cols,
             eval_set=(X_valid_cat,y_valid),
             use_best_model=True)

pred = cat_model.predict(X_valid_cat)
proba = cat_model.predict_proba(X_valid_cat)[:,1]

cat_f1 = f1_score(y_valid,pred)
cat_auc = roc_auc_score(y_valid,proba)


print("ğŸ”� ROC AUC:", cat_auc)
print("ğŸ�¯ F1 Score:", cat_f1)



data = {
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'ROC AUC': [xgb_auc, lgb_auc, cat_auc],
    'F1 Score': [xgb_f1, lgb_f1, cat_f1]
}
df_perf = pd.DataFrame(data)

# Plot line plot for ROC AUC and F1 Score
plt.figure(figsize=(8,5))
plt.plot(df_perf['Model'], df_perf['ROC AUC'], marker='o', label='ROC AUC')
plt.plot(df_perf['Model'], df_perf['F1 Score'], marker='s', label='F1 Score')

plt.ylim(0, 1)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.grid(True)
plt.legend()




X_full_data = df[num_cols + cat_cols].copy()
y_full_data = df.y.copy()

X_full_data[cat_cols] = ordinal_encoder.fit_transform(X_full_data[cat_cols])

final_model= lgb_model

final_model.fit(X_full_data,y_full_data,categorical_feature= cat_cols)


X_test = test_data[num_cols+cat_cols]
X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])


test_preds = final_model.predict_proba(X_test)[:,1]

submission = pd.DataFrame({
    'id' :test_data['id'],
    'y' : test_preds
})
submission.to_csv('submission.csv',index=False)

