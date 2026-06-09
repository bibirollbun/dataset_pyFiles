import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
features = [col for col in features if train[col].isnull().mean() < 0.4]  
features = [col for col in features if col != 'isFraud']

if 'TransactionID' in features:
    features.remove('TransactionID')


train[features] = train[features].apply(lambda col: col.fillna(col.median()))
test[features] = test[features].apply(lambda col: col.fillna(col.median()))


scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


X_train, X_val, y_train, y_val = train_test_split(train[features], train['isFraud'], test_size=0.2, random_state=42)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500
}


train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],  
    valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


predictions = model.predict(test[features])


submission = pd.DataFrame({'TransactionID': test['TransactionID'].astype(int),'isFraud': predictions})

submission.to_csv('submission.csv', index=False)




