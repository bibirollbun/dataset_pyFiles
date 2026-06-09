import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


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


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.005,
    'num_leaves': 128,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 4000
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_preds = np.zeros(len(test))
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train[features], train['isFraud'])):
    print(f"Fold {fold+1}")

    X_tr, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[train_idx]['isFraud'], train.iloc[val_idx]['isFraud']

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_preds = model.predict(X_val)
    auc = roc_auc_score(y_val, val_preds)
    auc_scores.append(auc)
    print(f" Fold {fold+1} AUC: {auc:.5f}")


    cv_preds += model.predict(test[features]) / skf.n_splits

print(f"\n Mean CV AUC: {np.mean(auc_scores):.5f}")


submission = pd.DataFrame({'TransactionID': test['TransactionID'].astype(int),'isFraud': cv_preds})
submission.to_csv('submission.csv', index=False)




