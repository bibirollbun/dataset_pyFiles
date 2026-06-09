import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)

from sklearn.linear_model import LogisticRegression


train = pd.read_csv('/kaggle/input/multi-class-prediction-of-cirrhosis-outcomess/train.csv')
test = pd.read_csv('/kaggle/input/multi-class-prediction-of-cirrhosis-outcomess/test.csv')
sample_sub = pd.read_csv('/kaggle/input/multi-class-prediction-of-cirrhosis-outcomess/sample_submission.csv')
train.drop(columns=["id"],axis=1,inplace=True)


print("Check Out Train DaTA Null Values: ",train.isnull().sum())
print("#"*130)
print(f"Train Data Shape: {train.shape}")
print("#"*130)
print(f"Train Data INFO: {train.info()}")
print("#"*130)


print("Check Out Test DaTA Null Values: ",test.isnull().sum())
print("#"*130)
print(f"Test Data Shape: {test.shape}")
print("#"*130)
print(f"Test Data INFO: {test.info()}")
print("#"*130)


train.head()


test.head()


def feature_engineer(df):
    for c in ['Bilirubin','Cholesterol','Copper','Alk_Phos','SGOT','Tryglicerides','Prothrombin']:
        df[f'log_{c}'] = np.log1p(df[c])
    df['Age'] = df['Age'] // 365
    df['is_male'] = (df['Sex'] == 'M').astype(int)
    df['edema_score'] = df['Edema'].map({'N':0, 'S':0.5, 'Y':1})
    df['risk'] = df['Bilirubin'] * df['Prothrombin']
    df['liver_health'] = df['Albumin'] / (df['log_Bilirubin'] + 0.1)
    bool_cols = ['Ascites','Hepatomegaly','Spiders']
    for c in bool_cols:
        df[c] = df[c].map({'N':0, 'Y':1})
    return df

train = feature_engineer(train)
test = feature_engineer(test)



train["Status"]=train['Status'].map({'C':0, 'CL':1, 'D':2})


cat_cols=["Drug","Sex","Edema"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


X = train.drop(['Status'], axis=1)
y = train['Status']
X_test = test.drop('id', axis=1)


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros((len(X), 3))
xgb_oof = np.zeros((len(X), 3))
lgb_preds = np.zeros((len(X_test), 3))
xgb_preds = np.zeros((len(X_test), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=70,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)
    lgb_preds += lgb_model.predict_proba(X_test) / n_splits
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=5,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X_val)
    xgb_preds += xgb_model.predict_proba(X_test) / n_splits


meta_X = np.hstack([lgb_oof, xgb_oof])
meta = LogisticRegression(multi_class='multinomial', max_iter=2000)
meta.fit(meta_X, y)

w1 = max(meta.coef_[0][:3].mean(), 0)
w2 = max(meta.coef_[0][3:].mean(), 0)
total = w1 + w2
w_lgb = w1 / total if total > 0 else 0.5
w_xgb = w2 / total if total > 0 else 0.5

final_preds = w_lgb * lgb_preds + w_xgb * xgb_preds
final_preds = np.clip(final_preds, 1e-15, 1-1e-15)
final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)

final_oof = w_lgb * lgb_oof + w_xgb * xgb_oof
print('Final blended CV:', log_loss(y, final_oof))
print(f'Blend → LGBM {w_lgb:.3f} | XGBoost {w_xgb:.3f}')


sub = sample_sub.copy()
sub[['Status_C','Status_CL','Status_D']] = final_preds
sub.to_csv('submission.csv', index=False)
sub.head()







