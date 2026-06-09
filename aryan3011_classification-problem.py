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


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone
import optuna
from sklearn.metrics import precision_recall_curve,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


test_df['education'].value_counts()


df['y'].value_counts()


corr = df.select_dtypes(exclude='object').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")


df.shape


df.head()


df.info()


df['month'].unique() # month will be encoded


# def Cyclic_Encoding(df):
#     month_mapping = {
#         'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
#         'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
#     }
    
#     df['month_numeric'] = df['month'].map(month_mapping)
#     df['month_sin'] = np.sin(2 * np.pi * df['month_numeric'] / 12)
#     df['month_cos'] = np.cos(2 * np.pi * df['month_numeric'] / 12)

# Cyclic_Encoding(X)
# Cyclic_Encoding(test_df)


df['age'].value_counts().sort_values(ascending=False).head(5)


df['age'].min(),df['age'].max()


df['age'].unique()


# df['age_*_balance']= df['age']* df['balance'] 


df['job'].value_counts()


(df.groupby('job')['y'].mean()*100).sort_values(ascending=False)


# df['job-Campaign'] = df['job']* df['campaign'] 
# #df['job-duration'] = df['job']* df['duration']
# #df['job-loan'] = df['job']* df['loan']
# #df['job-balance'] = df['job']* df['balance']


df['marital'].value_counts().sort_values(ascending=False)


df.groupby('marital')['y'].mean()


# df['marital-balance'] = df['marital'] * df['balance'] overfitting 


df.groupby('education')['y'].mean()


# education_order = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 2.5}
# df['education_ordinal'] = df['education'].map(education_order)

# df['education_age_interaction'] = df['education_ordinal'] * df['age']
# df['education_balance_ratio'] = df['education_ordinal'] * df['balance'] / 1000 did not work well



df['default'] = df['default'].map({'no': 0, 'yes': 1})
test_df['default'] = test_df['default'].map({'no': 0, 'yes': 1})


df['default'].value_counts()


df['default'].unique()


df.groupby('y')['default'].mean()


df[(df['balance'] <0) ].shape


df[(df['balance'] <0) & (df['y']==1)].shape


df['housing'].unique()


df['loan'].unique()


df['contact'].unique()


df['day'].unique()


df['duration'].unique()


df['campaign'].value_counts()


df['pdays'].value_counts()


df['previous'].value_counts()


df['poutcome'].value_counts()


def add_features(df,train_df=None):  
    df = df.copy()
# Ordinal encoding for education
    education_order = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 2.5}
    df['education_ordinal'] = df['education'].map(education_order)

# Interaction with age
    df['education_age_interaction'] = df['education_ordinal'] * df['age']

# Replace invalid pdays/campaign for safe ratios
    df['not_contacted_before'] = df['pdays'].replace(-1, np.nan)

# Ratio features
    df['balance_per_age'] = df['balance'] / (df['age'] + 1) #not performed well 
    df['campaign_per_pdays'] = df['campaign'] / (df['not_contacted_before'] + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    
    df['recent_campaign'] = df['pdays'].apply(lambda x: 1 if x < 10 else 0)

    df['is_weekend'] = df['day'].apply(lambda x: 1 if x in [6,7] else 0) 
    df['is_overdrawn'] = (df['balance'] < 0).astype(int)

    df['day_sin'] = np.sin(2*np.pi*df['day']/31)
    df['day_cos'] = np.cos(2*np.pi*df['day']/31)
    return df

df = add_features(df)
test_df = add_features(test_df, train_df=df)
# 0.96571 V11 with job age and education feature
# 0.96582 with job and age feature
# 0.96553 with age feature is present
df = df.drop(columns=['id','day',"education"], errors='ignore')
test_df = test_df.drop(columns=['day',"education"], errors='ignore')


df


X = df.drop('y',axis=1)
y=df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


cat_col = X.select_dtypes(include=['object', 'category']).drop(columns=['month','job'])
num_col = X.select_dtypes(exclude=['object', 'category'])
cat  = cat_col.columns.to_list()
num = num_col.columns.to_list()


cat


num


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',sparse_output=False), cat),
        ('TE',TargetEncoder(),['month','job']
        )
    ],
    remainder='passthrough'  
).set_output(transform="pandas")


# cat_pipeline = Pipeline(steps=[
# ('preprocessor', preprocessor), 
# ('classifier', CatBoostClassifier(random_state=42,verbose=100 )
#  )
# ])

# # Fit the pipeline
# cat_pipeline.fit(X_train, y_train)

# X_test_pipe = test_df.drop('id', axis=1)

# y_probs = cat_pipeline.predict_proba(X_test_pipe)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],  
#     "y": y_probs   
# })
# submission.to_csv('submission.csv', index=False)
# 0.96753 V26
# 0.96768 V27


# y_pred = cat_pipeline.predict(X_test)
# y_val_probs = cat_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)



# light_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),    
#     ('classifier', lgb.LGBMClassifier(random_state=42))
    
# ])
# light_pipeline.fit(X_train, y_train)
# X_test_pipe = test_df.drop('id', axis=1)
# y_probs = light_pipeline.predict_proba(X_test_pipe)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],  
#     "y": y_probs   
# })
# submission.to_csv('submission.csv', index=False)
# # 0.96553 V28


# y_pred = light_pipeline.predict(X_test)
# y_val_probs = light_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)
# Validation ROC AUC Score: 0.9646484216064709
# Validation ROC AUC Score: 0.9646841839119557


# xgb_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor), 
#     ('classifier', XGBClassifier(random_state=42),
#     )
    
# ])

# xgb_pipeline.fit(X_train, y_train)

# y_probs = xgb_pipeline.predict_proba(test_df)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],  
#     "y": y_probs   
# })
# submission.to_csv('submission.csv', index=False)
# # 0.96653 V29


# y_pred = xgb_pipeline.predict(X_test)
# y_val_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)
# Validation ROC AUC Score: 0.9658272175149378
# 0.9660071362273847 when day sin cos and overdrawn is commented 
# Validation ROC AUC Score: 0.9656857941649266


# def objective(trial):
#     """Objective function for Optuna with Stratified K-Fold CV."""

#     # Define the hyperparameter search space
#     params = {
#         'task_type': 'GPU',
#         'iterations': trial.suggest_int('iterations', 1000, 3000),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
#         'border_count': trial.suggest_int('border_count', 32, 455),
#         'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'od_type': 'Iter',
#         'od_wait': trial.suggest_int('od_wait', 50, 200),
#         'random_state': 42,
#         'verbose': 0
#     }

#     # Stratified K-Fold setup
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []

#     for train_idx, val_idx in skf.split(X_train, y_train):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         # Create pipeline
#         cat_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('classifier', CatBoostClassifier(**params))
#         ])

#         # Fit and evaluate
#         cat_pipeline.fit(X_tr, y_tr)
#         y_val_probs = cat_pipeline.predict_proba(X_val)[:, 1]
#         aucs.append(roc_auc_score(y_val, y_val_probs))

#     # Return mean AUC across folds
#     return np.mean(aucs)

# # --- Run the Optuna Study ---
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print("Hyperparameter tuning finished.")
# print("Best trial found:")
# best_trial = study.best_trial
# print(f"\n  Best ROC AUC Score: {best_trial.value:.6f}")
# print("\n  Best Hyperparameters:")
# for key, value in best_trial.params.items():
#     print(f"    '{key}': {value}")


# cat_pipeline = Pipeline(steps=[
# ('preprocessor', preprocessor), 
# ('classifier', CatBoostClassifier(random_state=42, iterations = 2868 , depth = 6 ,
#     learning_rate = 0.18469288373640855,
#     l2_leaf_reg = 8.580107939074999,
#     border_count = 455,
#     random_strength = 0.28034816316458594 ,
#     bagging_temperature = 0.02295693714601682, 
#     od_wait = 104,
#     verbose =200 )
# )
# ])


# cat_pipeline.fit(X_train, y_train)
# X_test_pipe = test_df.drop('id', axis=1)

# y_probs = cat_pipeline.predict_proba(X_test_pipe)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],  
#     "y": y_probs   
# })

# submission.to_csv('submission.csv', index=False)

# # # 0.96929


# y_pred = cat_pipeline.predict(X_test)
# y_val_probs = cat_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)
# Validation ROC AUC Score: 0.9678553133195555


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 1000, 2500),  # Very narrow
#         "max_depth": trial.suggest_int("max_depth", 5, 8),  # Around CatBoost's 6
#         "num_leaves": trial.suggest_int("num_leaves", 31, 100),  # Smaller
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),  # Around CatBoost's 0.18
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 40),
#         "subsample": trial.suggest_float("subsample", 0.85, 0.95),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.85, 0.95),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 15.0),  # Around CatBoost's 8.5
#         "random_state": 42,
#         "n_jobs": 1,
#         "objective": "binary",
#         "device_type": "gpu",
#         "metric": "auc",
#         "verbosity": -1,
#         "early_stopping_rounds": 100  # Aggressive early stopping
#     }
    
#     # Use 2-fold CV for even faster trials
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     aucs = []
    
#     for train_idx, val_idx in skf.split(X_train, y_train):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
#         X_tr_transformed = preprocessor.fit_transform(X_tr, y_tr)
#         X_val_transformed = preprocessor.transform(X_val)
        
#         model = lgb.LGBMClassifier(**params)
#         model.fit(
#             X_tr_transformed, y_tr,
#             eval_set=[(X_val_transformed, y_val)],
#             callbacks=[lgb.early_stopping(100, verbose=False)]
#         )
        
#         y_val_probs = model.predict_proba(X_val_transformed)[:, 1]
#         aucs.append(roc_auc_score(y_val, y_val_probs))
    
#     return np.mean(aucs)

# # Aggressive pruning
# study = optuna.create_study(
#     direction="maximize",
#     pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
# )
# study.optimize(objective, n_trials=70, show_progress_bar=True)


# light_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),    
#     ('classifier', lgb.LGBMClassifier(
# random_state=42,
# n_estimators = 1882,
# max_depth  = 15, 
# num_leaves = 181, 
# learning_rate = 0.020961333927623275, 
# min_child_samples = 52, 
# subsample = 0.6775483862329711, 
# colsample_bytree = 0.5213804801393623, 
# reg_alpha = 2.115676119115116, 
# reg_lambda = 1.0222328760790468e-06))   
# ])

# light_pipeline.fit(X_train, y_train)
# X_test_pipe = test_df.drop('id', axis=1)
# y_probs = light_pipeline.predict_proba(X_test_pipe)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],  
#     "y": y_probs   
# })
# submission.to_csv('submission.csv', index=False)
# # 0.96974 V32


# y_pred = light_pipeline.predict(X_test)
# y_val_probs = light_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)
# Validation ROC AUC Score: 0.9688210420669978


# def objective(trial):
#     """Objective function for Optuna to optimize XGBoost hyperparameters with StratifiedKFold CV + early stopping."""

#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 500, 6000),   # wider
#         'max_depth': trial.suggest_int('max_depth', 2, 20),             # allow deeper
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True), # wider, log scale
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),        
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 10),                  # wider
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 50), # much wider
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True), # L1 reg
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True), # L2 reg
#         'random_state': 42,
#         "tree_method": "hist",
#         "early_stopping_rounds" :100,
#         "device": "cuda",   # use GPU
#         'eval_metric': 'auc'
#     }
    
#     # Stratified K-Fold CV
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     aucs = []
#     best_iters = []

#     for train_idx, valid_idx in skf.split(X_train, y_train):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
#         # Preprocess (need to pass y_tr if encoder requires it)
#         X_tr = preprocessor.fit_transform(X_tr, y_tr)
#         X_val = preprocessor.transform(X_val)
    
#         # Initialize XGBoost
#         model = XGBClassifier(**params)
    
#         # Fit with early stopping
#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
    
#             verbose=False
#         )
    
#         # Predict
#         y_val_probs = model.predict_proba(X_val)[:, 1]
    
#         # Store AUC
#         aucs.append(roc_auc_score(y_val, y_val_probs))
    
#         # Store best iteration
#         best_iters.append(model.get_booster().best_iteration)

#     # Average AUC across folds
#     mean_auc = np.mean(aucs)

#     # Log the mean of best iterations so you can reuse later
#     trial.set_user_attr("avg_best_iter", int(np.mean(best_iters)))

#     return mean_auc


# # --- 2. Run Optuna Study ---
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=75, show_progress_bar=True)

# # --- 3. Best Results ---
# print("Best ROC AUC:", study.best_trial.value)
# print("Best Params:", study.best_trial.params)
# print("Best avg n_estimators:", study.best_trial.user_attrs["avg_best_iter"])



# final_xgb_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(
#     n_estimators = 2082,
#     max_depth = 10,
#     learning_rate = 0.015924390204725847,
#     subsample = 0.7304173937626887,
#     colsample_bytree =  0.6008751697408913, 
#     gamma = 0.42247912487219413,
#     min_child_weight = 11, 
#     reg_alpha = 0.7974172737095002,
#     reg_lambda = 0.19197122039242795))
# ])

# final_xgb_pipeline.fit(X, y)

# X_test_pipe = test_df.drop('id', axis=1)
# y_probs = final_xgb_pipeline.predict_proba(X_test_pipe)[:, 1]

# submission = pd.DataFrame({
#     "id": test_df['id'],
#     "y": y_probs
# })
# submission.to_csv("submission.csv", index=False)
# 0.96926


# y_pred = final_xgb_pipeline.predict(X_test)
# y_val_probs = final_xgb_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)


# # Fit both models
# cat_pipeline.fit(X_train, y_train)
# light_pipeline.fit(X_train, y_train)

# # Predict
# X_test_pipe = test_df.drop('id', axis=1)
# cat_probs = cat_pipeline.predict_proba(X_test_pipe)[:, 1]
# light_probs = light_pipeline.predict_proba(X_test_pipe)[:, 1]

# # Simple average (bagging)
# y_probs_avg = (cat_probs + light_probs) / 2  

# # Or weighted average if one model is stronger
# # y_probs_avg = 0.6 * light_probs + 0.4 * cat_probs  

# # Submission
# submission = pd.DataFrame({
#     "id": test_df['id'],
#     "y": y_probs_avg
# })
# submission.to_csv("submission.csv", index=False)
# # 0.97010 V34 when simple averaging is used


# estimators = [
#     ('xgb', XGBClassifier(
#        n_estimators = 2082,
#     max_depth = 8,
#     learning_rate = 0.015924390204725847,
#     subsample = 0.7304173937626887,
#     colsample_bytree =  0.6008751697408913, 
#     gamma = 0.42247912487219413,
#     min_child_weight = 11, 
#     reg_alpha = 0.7974172737095002,
#     reg_lambda = 0.19197122039242795
#     )),
#     ('lgbm', lgb.LGBMClassifier(
#         random_state=42,
# n_estimators = 1882,
# max_depth  = 12, 
# num_leaves = 181, 
# learning_rate = 0.020961333927623275, 
# min_child_samples = 52, 
# subsample = 0.6775483862329711, 
# colsample_bytree = 0.5213804801393623, 
# reg_alpha = 2.115676119115116, 
# reg_lambda = 1.0222328760790468e-06
#     ))
# ]

# # Stacking model
# stack = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegressionCV(
#         penalty='l2',
#         Cs=10,        # 10 candidate regularization strengths
#         cv=5,         # inner CV for tuning regularization
#         max_iter=2000,
#         scoring="roc_auc",
#         n_jobs=-1
#     ),
#     cv=5,            # outer CV for stacking
#     n_jobs=-1
# )
# # Pipeline with preprocessing
# stack_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('stack', stack)
# ])

# # Fit
# stack_pipeline.fit(X_train, y_train)

# # Predict
# X_test_pipe = test_df.drop('id', axis=1)
# y_probs_stack = stack_pipeline.predict_proba(X_test_pipe)[:, 1]

# # Submission
# submission = pd.DataFrame({
#     "id": test_df['id'],
#     "y": y_probs_stack
# })
# submission.to_csv("submission.csv", index=False)


# y_pred = stack_pipeline.predict(X_test)
# y_val_probs = stack_pipeline.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_val_probs)
# print("Validation ROC AUC Score:", roc_auc)


xgb_model = XGBClassifier(
    n_estimators=2082,
    max_depth=8,
    learning_rate=0.015924390204725847,
    subsample=0.7304173937626887,
    colsample_bytree=0.6008751697408913,
    gamma=0.42247912487219413,
    min_child_weight=11,
    reg_alpha=0.7974172737095002,
    reg_lambda=0.19197122039242795,
    random_state=42,
    early_stopping_rounds=100,
    use_label_encoder=False,
    eval_metric="auc"
)

lgb_model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=1882,
    max_depth=12,
    num_leaves=181,
    learning_rate=0.020961333927623275,
    min_child_samples=52,
    subsample=0.6775483862329711,
    colsample_bytree=0.5213804801393623,
    reg_alpha=2.115676119115116,
    early_stopping_rounds=100,
    reg_lambda=1.0222328760790468e-06
)

# OOF containers
oof_preds = np.zeros((len(X_train), 2))   # 2 base models
test_preds = np.zeros((len(test_df), 2))
cv_scores = []

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_test_pipe = test_df.drop("id", axis=1)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFOLD {fold+1}")
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # Preprocessing
    X_tr_pipe = preprocessor.fit_transform(X_tr, y_tr)
    X_va_pipe = preprocessor.transform(X_va)
    X_test_pipe_proc = preprocessor.transform(X_test_pipe)

    # XGB
    xgb_model.fit(
        X_tr_pipe, y_tr,
        eval_set=[(X_va_pipe, y_va)],
    )
    oof_preds[va_idx, 0] = xgb_model.predict_proba(X_va_pipe)[:, 1]
    test_preds[:, 0] += xgb_model.predict_proba(X_test_pipe_proc)[:, 1] / skf.n_splits

    # LGBM
    lgb_model.fit(
        X_tr_pipe, y_tr,
        eval_set=[(X_va_pipe, y_va)],
    )
    oof_preds[va_idx, 1] = lgb_model.predict_proba(X_va_pipe)[:, 1]
    test_preds[:, 1] += lgb_model.predict_proba(X_test_pipe_proc)[:, 1] / skf.n_splits

    # Fold AUC of averaged base models
    fold_auc = roc_auc_score(y_va, 0.5*oof_preds[va_idx,0] + 0.5*oof_preds[va_idx,1])
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

# Meta-learner
meta = LogisticRegressionCV(
    penalty="l2",
    Cs=10,
    cv=5,
    scoring="roc_auc",
    max_iter=2000,
    n_jobs=-1
)
meta.fit(oof_preds, y_train)

# OOF AUC from meta
oof_auc = roc_auc_score(y_train, meta.predict_proba(oof_preds)[:, 1])
print("\nOOF Meta AUC:", round(oof_auc, 5))
print("Mean base blend CV AUC:", np.mean(cv_scores))

# Final test prediction
final_test_preds = meta.predict_proba(test_preds)[:, 1]

# Submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "y": final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved: submission.csv")


# Validation ROC AUC Score: 0.9646879876853851 with no interaction feature
# Validation ROC AUC Score: 0.9647357481421438 when age * balance
# Validation ROC AUC Score: 0.9646698649024839 Ordinal encoding on job
# Validation ROC AUC Score: 0.9646414763283874 TE on job 
#ROC AUC        : 0.9647 polynomial feature on age had no impact 
# Validation ROC AUC Score: 0.9643087219036823 cyclic encoding to month and day
# ROC AUC        : 0.9646 when job*duration is added Validation ROC AUC Score: 0.9647357481421438 when job*duration is removed
# Validation ROC AUC Score: 0.9647412492658491 when job* campaign is added a very slight small margin increase 
# Validation ROC AUC Score: 0.9647551773325195  when education_ordinal * age 
# Validation ROC AUC Score: 0.9656854148920596 when pdays -1 is replaced with mean and job's 0 with mean
# Validation ROC AUC Score: 0.965744609800584 when p days -1 replaced with nan 
# Validation ROC AUC Score: 0.9657668654073855 when campaign/day is introduced 
# Validation ROC AUC Score: 0.9658203480914277 when duration/campaign is introduced
# Validation ROC AUC Score: 0.9659246706361602 ADDED RECENt campaign 
# Validation ROC AUC Score: 0.9659515375341714 added job to TE
# Validation ROC AUC Score: 0.9666969139556669 with catboost

# Validation ROC AUC Score: 0.9687449928557599 with tuned catboost using optuna



# # Get trained model
# model = pipeline.named_steps['classifier']

# # Get feature importances
# importances = pipeline.feature_importances_

# # Get feature names after preprocessing
# feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# # Put into DataFrame
# fi_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': importances
# }).sort_values(by='importance', ascending=False)

# print(fi_df.head(40))


