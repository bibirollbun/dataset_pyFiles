import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_train.head()


df_train.shape


df_train.info()


df_train.isnull().sum()


df_test.head()


df_test.shape


df_test.info()


df_test.isnull().sum()


df_train['Personality'].value_counts()


sns.countplot(x=df_train['Personality'], palette='pastel')
plt.title('Introvert vs Extrovert');


numeric = df_train.select_dtypes(include=['float64', 'int64']).columns.drop('id', errors='ignore').tolist()
categoric = df_train.select_dtypes(include=['object']).columns.drop('Personality', errors='ignore').tolist()

print(f"Numeric: {numeric}")
print(f"Categoric: {categoric}")


for col in numeric:
    sns.kdeplot(data=df_train, x=col, hue='Personality', fill=True, palette='pastel')
    plt.title(f"{col} Introvert vs Extrovert")
    plt.show()


for col in categoric:
    sns.countplot(data=df_train, x=col, hue='Personality', palette='pastel')
    plt.title(f"{col} vs Personality")
    plt.show()


train = df_train.copy()
test = df_test.copy()


map_dict = {'Yes': 1, 'No': 0}
train['Stage_fear'] = train['Stage_fear'].map(map_dict)
test['Stage_fear'] = test['Stage_fear'].map(map_dict)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(map_dict)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(map_dict)

map_personality = {'Introvert': 1, 'Extrovert': 0}
train['Personality'] = train['Personality'].map(map_personality)


train['Social_to_Alone_Ratio'] = train['Social_event_attendance'] / (train['Time_spent_Alone'] + 1)
test['Social_to_Alone_Ratio'] = test['Social_event_attendance'] / (test['Time_spent_Alone'] + 1)

train['Social_Volume'] = train['Friends_circle_size'] * train['Going_outside']
test['Social_Volume'] = test['Friends_circle_size'] * test['Going_outside']


x = train.drop(['Personality', 'id'], axis=1, errors='ignore') 
y = train['Personality']
test_df = test.drop(['id'], axis=1, errors='ignore') 


models = {"XGBoost": xgb.XGBClassifier(n_estimators=1000,learning_rate=0.05,random_state=42,n_jobs=-1,
                                       early_stopping_rounds=50,verbosity=0),
          "LightGBM": lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.05,random_state=42,n_jobs=-1,
                                         verbosity=-1),
          "CatBoost": CatBoostClassifier(n_estimators=1000,learning_rate=0.05,random_state=42,verbose=0,
                                         allow_writing_files=False)}
results = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():    
    fold_scores = []
    for train_idx, val_idx in skf.split(x, y):
        X_train_fold, y_train_fold = x.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = x.iloc[val_idx], y.iloc[val_idx]
        if name == "XGBoost":
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        elif name == "LightGBM":
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], 
                      eval_metric='binary_logloss', callbacks=callbacks)
        elif name == "CatBoost":
            model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, 
                      verbose=False)
            
        preds = model.predict(X_val_fold)
        fold_scores.append(accuracy_score(y_val_fold, preds))
    
    avg_score = np.mean(fold_scores)
    results[name] = avg_score
    print(f"{name}: {avg_score:.5f}")


# Optuna
def objective(trial):
    param = {'objective': 'binary','metric': 'binary_logloss','verbosity': -1,'boosting_type': 'gbdt','random_state': 42,
             'n_estimators': 1000,'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
             'num_leaves': trial.suggest_int('num_leaves', 20, 150),
             'max_depth': trial.suggest_int('max_depth', 3, 12),
             'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
             'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),}
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    model = lgb.LGBMClassifier(**param)  
    for train_idx, val_idx in skf.split(x, y):
        X_train_fold, X_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold,eval_set=[(X_val_fold, y_val_fold)],eval_metric='binary_logloss',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])   
        preds = model.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, preds))    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) 

print(f"{study.best_value:.5f}")
print(study.best_params)


lgbm_params = {'objective': 'binary','metric': 'binary_logloss','verbosity': -1,'boosting_type': 'gbdt',
               'random_state': 42,'n_estimators': 1000,'learning_rate': 0.08502962621721107,'num_leaves': 94,
               'max_depth': 6,'min_child_samples': 88,'subsample': 0.6397617433185914,
               'colsample_bytree': 0.6492257609378472,'reg_alpha': 3.13530274125673,'reg_lambda': 0.08461684522777535}
clf_lgbm = lgb.LGBMClassifier(**lgbm_params)

# XGBoost
clf_xgb = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.05,max_depth=6,random_state=42,n_jobs=-1,verbosity=0)
# CatBoost
clf_cat = CatBoostClassifier(n_estimators=1000,learning_rate=0.05,verbose=0,random_state=42,allow_writing_files=False)

# Ensemble Voting
voting_clf = VotingClassifier(estimators=[('lgbm', clf_lgbm), ('xgb', clf_xgb), ('cat', clf_cat)],voting='soft')

voting_clf.fit(x, y)
final_preds = voting_clf.predict(test_df)


submission = pd.DataFrame({'id': df_test['id'],'Personality': final_preds})
reverse_map = {1: 'Introvert', 0: 'Extrovert'}
submission['Personality'] = submission['Personality'].map(reverse_map)
submission.to_csv('submission.csv', index=False)


import joblib
joblib.dump(voting_clf, 'introvert_model.joblib')

