import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
for col in categorical_cols:
    for df in [train, test]:
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))


features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
features = [col for col in features if train[col].isnull().mean() < 0.4]
features = [col for col in features if col not in ['isFraud', 'TransactionID']]
features = list(set(features + categorical_cols))



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


model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.01,
    n_estimators=2000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method='hist',
    use_label_encoder=False,
    verbosity=1,
    early_stopping_rounds=50
)



model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)


predictions = model.predict_proba(test[features])[:, 1]


submission = pd.DataFrame({
    'TransactionID': test['TransactionID'].astype(int),
    'isFraud': predictions
})
submission.to_csv('submission_xgboost.csv', index=False)




