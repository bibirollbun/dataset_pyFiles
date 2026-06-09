!pip uninstall -y imbalanced-learn
!pip install imbalanced-learn==0.10.1


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
for col in categorical_cols:
    for df in [train, test]:
        df[col] = df[col].fillna('unknown')  
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]) 


features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
features = [col for col in features if train[col].isnull().mean() < 0.4] 
features = [col for col in features if col != 'isFraud']  
for col in categorical_cols:
    if col in features:
        features.remove(col)  
features += categorical_cols  


if 'TransactionID' in features:
    features.remove('TransactionID')


for df in [train, test]:
    for col in features:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())  
        else:
            df[col] = df[col].fillna('unknown')


scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


X_train, X_val, y_train, y_val = train_test_split(train[features], train['isFraud'], test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 128,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000
}


train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


predictions = model.predict(test[features])


submission = pd.DataFrame({'TransactionID': test['TransactionID'].astype(int), 'isFraud': predictions})
submission.to_csv('submission.csv', index=False)




