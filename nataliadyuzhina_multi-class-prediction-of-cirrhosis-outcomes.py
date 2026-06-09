pip install lazypredict


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
import lightgbm as lgb
from lazypredict.Supervised import LazyClassifier, accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


# Loading the dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s3e26/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e26/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s3e26/sample_submission.csv')


df_train.head()



df_test.head()



df_sub.head()



df_train.shape,df_test.shape,df_sub.shape



df_train.isnull().sum()



df_test.isnull().sum()



df_train.dtypes,df_test.dtypes



df_train.describe()



df_train = df_train.drop(['id'],axis=1)
df_test = df_test.drop(['id'],axis=1)


df_train.dtypes


df_train['Status'].value_counts()


df_test.dtypes


from sklearn.preprocessing import LabelEncoder


# Identify object-type columns
object_columns = df_train.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert object-type columns to numerical
for col in object_columns:
    df_train[col] = label_encoder.fit_transform(df_train[col])


df_train.dtypes


df_train['Status'].value_counts()



# Identify object-type columns
object_columns1 = df_test.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert object-type columns to numerical
for col in object_columns1:
    df_test[col] = label_encoder.fit_transform(df_test[col])


df_test.dtypes


df_train.head()


# Visualize the distribution of the target column
plt.figure(figsize=(6, 6))
sns.countplot(x='Status', data=df_train)
plt.title('Distribution of Target Column (status)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(14,10))
corr=df_train.corr()
sns.heatmap(corr,annot=True,cmap='mako',mask=np.triu(corr))
plt.show()


X = df_train.drop(['Status'], axis = 1)
y = df_train['Status']


# splitting data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


print("Length of X_train:", len(X_train))
print("Length of X_valid:", len(X_valid))
print("Length of y_train:", len(y_train))
print("Length of y_valid:", len(y_valid))


# let's try lazyregressor
reg = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
models


import xgboost as xgb
x = xgb.XGBClassifier(objective='multi:softmax', num_leaves=10)


import optuna
import lightgbm as lgb


lgbm_params = {'objective': 'multi_logloss', 
               'max_depth': 9, 'min_child_samples': 14, 
               'learning_rate': 0.034869481921747415, 
               'n_estimators': 274, 'min_child_weight': 9, 
               'subsample': 0.7717873512945741, 
               'colsample_bytree': 0.1702910221565107, 
               'reg_alpha': 0.10626128775335533, 
               'reg_lambda': 0.624196407787772, 
               'random_state': 42}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)
lgbm_model.fit(X_train, y_train)
y_pred_proba = lgbm_model.predict_proba(X_valid)
loss = log_loss(y_valid, y_pred_proba)
print('Loss:', loss)


def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(set(y_train)),
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
        'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 10),
        'subsample': trial.suggest_uniform('subsample', 0.0, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_valid)
    loss = log_loss(y_valid, y_pred_proba)
    print('Loss:', loss)
    return loss

run=0

if run==1:

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print('Best trial:')
    trial = study.best_trial

    print('Value: {}'.format(trial.value))
    print('Params: ')
    for key, value in trial.params.items():
        print(' {}: {}'.format(key, value))


params = {'booster': 'dart',
 'lambda': 0.0006919381888069279,
 'alpha': 0.35665520735590184,
 'max_depth': 10,
 'eta': 0.045477449903818745,
 'gamma': 2.0716859908951374e-06,
 'colsample_bytree': 0.17933987267073742,
 'min_child_weight': 4.5831199957843145,
 'subsample': 0.3505025486528871,
 'n_estimators': 190}


x = xgb.XGBClassifier(**params)


x.fit(X_train,y_train)


pred = lgbm_model.predict_proba(X_valid)


log_loss(y_valid,pred)


lgbm_best_param = {'random_state': 42, 'n_estimators': 850, 'max_depth': 24, 
                   'learning_rate': 0.010057071900335017, 'reg_alpha': 0.9880777014859632, 
                   'reg_lambda': 0.9973409467621859, 'min_child_weight': 0.06493159533714984, 
                   'min_child_samples': 12, 'subsample': 0.6797069674560879, 
                   'subsample_freq': 2, 'colsample_bytree': 0.16452242442196183, 'num_leaves': 94}

final_model = lgb.LGBMClassifier(**lgbm_best_param)
final_model.fit(X_train, y_train)
print('Train Error:', log_loss(y_valid, final_model.predict_proba(X_valid)))


y_probs = final_model.predict_proba(df_test)


df_sub['Status_C'] = y_probs[:, 0]
df_sub['Status_CL'] = y_probs[:, 1]
df_sub['Status_D'] = y_probs[:, 2]


df_sub.to_csv('submission.csv',index=False)


df_sub.head()




