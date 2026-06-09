import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
import optuna
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


train_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


test_df.head()


test_df.shape


test_df.info()


test_df.isnull().sum()


sns.countplot(x=train_df['class'], palette='pastel')
plt.title('Distribution of Target Variable (e: Edible, p: Poisonous)')
plt.show()

print(train_df['class'].value_counts(normalize=True))


categorical = [col for col in train_df.columns if train_df[col].dtype == 'object' and col not in ['class']]
numeric = ['cap-diameter','stem-height','stem-width']

#Distribution of numerical variables
train_df[numeric].hist(bins=30, figsize=(15, 10), color='royalblue')
plt.show()
#Outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric):
    plt.subplot(3, 3, i+1) 
    sns.boxplot(x='class', y=col, data=train_df, palette='pastel')
plt.tight_layout()
plt.show()
#Poisonous rates
for col in categorical[:6]: #First 6 columns
    plt.figure(figsize=(20, 5))
    sns.countplot(x=col, hue='class', data=train_df, palette='pastel')
    plt.title(f'{col} - Class Distribution')
    plt.xticks(rotation=90)
    plt.show()


#If more than 95% of a column is missing or contains only one type of value:Unnecessary Column
cols_to_drop = []
for col in train_df.columns:
    if train_df[col].nunique() <= 1:
        cols_to_drop.append(col)
print(f"Unnecessary Columns: {cols_to_drop}")
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

#Categorical/Numerical Column List
cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' and col != 'class']
num_cols = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

#Imputation
#Missing
for col in cat_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col] = test_df[col].fillna('Missing')
#Median
num_imputer = SimpleImputer(strategy='median')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

#Rare Labels
#Other
def handle_rare_labels(df_train, df_test, column, threshold=0.01):
    counts = df_train[column].value_counts(normalize=True)
    valid_labels = counts[counts >= threshold].index
    
    df_train[column] = df_train[column].apply(lambda x: x if x in valid_labels else 'Other')
    df_test[column] = df_test[column].apply(lambda x: x if x in valid_labels else 'Other')
    return df_train, df_test

for col in cat_cols:
    train_df, test_df = handle_rare_labels(train_df, test_df, col, threshold=0.01)

#Label Encoding
le = LabelEncoder()
for col in cat_cols:
    full_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(full_data)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
train_df['class'] = train_df['class'].map({'e': 0, 'p': 1})


train_df.head()


test_df.head()


x = train_df.drop(['class', 'id'], axis=1, errors='ignore') 
y = train_df['class']


X_train, X_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y) #Stratified Sampling


#XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,
                              random_state=42,n_jobs=-1,early_stopping_rounds=50,eval_metric="logloss")

xgb_model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=False)

y_pred_xgb = xgb_model.predict(X_val)
mcc_xgb = matthews_corrcoef(y_val, y_pred_xgb)
print(f"XGBoost MCC: {mcc_xgb:.5f}")

#LightGBM 
lgb_model = lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.05,num_leaves=31,random_state=42,n_jobs=-1,verbose=-1)

callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

lgb_model.fit(X_train, y_train,eval_set=[(X_val, y_val)],eval_metric='binary_logloss',callbacks=callbacks)

y_pred_lgb = lgb_model.predict(X_val)
mcc_lgb = matthews_corrcoef(y_val, y_pred_lgb)
print(f"LightGBM MCC: {mcc_lgb:.5f}")


X_test = test_df.drop(['id'], axis=1, errors='ignore')
test_preds = xgb_model.predict(X_test)

inv_map = {0: 'e', 1: 'p'}
final_preds = [inv_map[x] for x in test_preds]


submission = pd.DataFrame({'id': test_df['id'],'class': final_preds})
submission.to_csv('submission.csv', index=False)


optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)}
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=False)
    
    preds = model.predict(X_val)
    score = matthews_corrcoef(y_val, preds)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

print(f"The best value: {study.best_value:.5f}")
print(study.best_params)


best_params = study.best_params

best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'logloss'
best_params['tree_method'] = 'gpu_hist'
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

final_preds = final_model.predict(X_test)
final_preds_labels = [inv_map[x] for x in final_preds] 


submission_optuna = pd.DataFrame({'id': test_df['id'],'class': final_preds_labels})
submission_optuna.to_csv('submission_v2.csv', index=False)


import joblib

artifacts = {
    'model': final_model,'num_imputer': num_imputer,'cat_cols': cat_cols,'num_cols': num_cols,
    'encoders': {},'valid_labels': {}}

raw_train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv').drop(columns=cols_to_drop) 

for col in cat_cols:
    counts = raw_train[col].fillna('Missing').value_counts(normalize=True)
    valid = counts[counts >= 0.01].index.tolist()
    artifacts['valid_labels'][col] = valid
    
    possible_values = valid + ['Other', 'Missing']
    le = LabelEncoder()
    le.fit(possible_values)
    artifacts['encoders'][col] = le

joblib.dump(artifacts, 'mushroom_pipeline.pkl')

