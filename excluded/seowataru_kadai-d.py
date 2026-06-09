import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score


train = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")

##BMI
#for df in [train, test]:
#    df['BMI'] = df['weight(kg)'] / ((df['height(cm)']/100) ** 2)

##Gtp*hemoglobin
#for df in [train, test]:
#    df['Gtp*hemoglobin'] = df['Gtp']*df['hemoglobin'] 

##waist/height
#for df in [train, test]:
#    df['waist/height'] = df['waist(cm)'] / df['height(cm)']

##LDL/HDL
#for df in[train,test]:
#    df['LDL/HDL']=df['LDL']/df['HDL']

##height*hemoglobin
for df in[train,test]:
    df['height*hemoglobin']=df['height(cm)']*df['hemoglobin']



str_cols = train.select_dtypes(include=['object']).columns.tolist()

feature_cols = [col for col in train.columns if col not in ['id', 'smoking'] ]

num_cols = [col for col in feature_cols if col not in str_cols]

train[str_cols] = train[str_cols].fillna('NA')
test[str_cols] = test[str_cols].fillna('NA')


for col in str_cols:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


train_x = train[feature_cols]
train_y = train['smoking']
test_x = test[feature_cols]


kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]


dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 71}
num_round = 50

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist)

va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
auc = roc_auc_score(va_y, va_pred)
print(f'logloss: {score:.4f}')
print(f'AUC: {auc:.4f}')

pred = model.predict(dtest)

submission = pd.DataFrame({
    'id': test['id'],
    'smoking': pred
})
submission.to_csv('submission.csv', index=False)


