import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,  StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder , OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as  xgb
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib


train_path = '/kaggle/input/playground-series-s5e8/train.csv'
test_path = '/kaggle/input/playground-series-s5e8/test.csv'
submission_path = '/kaggle/input/playground-series-s5e8/sample_submission.csv'
original = '/kaggle/input/bank-marketing-dataset-full'

target = 'y'
n_folds = 5
seed = 5

cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state =seed, )
metirx = roc_auc_score
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


train.head()


test.select_dtypes(include=['number']).columns.tolist


iqr_lst = ['duration','campaign','pdays','previous']
def iqr_ting(df, list_):
    mask = np.ones(len(df), dtype=bool) 
    for i in list_:
        q1 = np.quantile(df[i], 0.25)
        q3 = np.quantile(df[i], 0.75)
        iqr = q3-q1
        lower = q1 - 1.5 * iqr
        higher = q3 + 1.5 * iqr
        mask &= (df[i] > lower) & (df[i] < higher)
        return df[mask]
train = iqr_ting(train, iqr_lst)   
test = iqr_ting(test, iqr_lst)


# Columns to apply ordinal encoder?
'''education
job as people with better job will have higher chance of buying the subscription
poutcome 
'''
# binary encoding 
''' default , loan and housing as they have only 2 values yes and no'''

# Cyclic encoding ?
''' days and months as we want to preserve their nature and preodic cyclic'''


oe_education = OrdinalEncoder(categories=[['secondary', 'primary', 'tertiary', 'unknown']])
train['education'] = oe_education.fit_transform(train[['education']])
test['education'] = oe_education.transform(test[['education']])

oe_job = OrdinalEncoder(categories=[[
    'services', 'blue-collar', 'technician', 'admin.', 'housemaid',
    'entrepreneur', 'management', 'unemployed', 'self-employed',
    'student', 'retired', 'unknown'
]])
train['job'] = oe_job.fit_transform(train[['job']])
test['job'] = oe_job.transform(test[['job']])

for col in ['default', 'loan', 'housing']:
    train[col] = train[col].map({'yes': 1, 'no': 0})
    test[col] = test[col].map({'yes': 1, 'no': 0})

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

train['month'] = train['month'].map(month_map)
test['month'] = test['month'].map(month_map)

for df in [train, test]:
    df['cyclic_sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cyclic_cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['cyclic_sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cyclic_cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dept'] = df['balance'].apply(lambda x: 1 if x >= 0 else 0)
    df.drop(columns=['month', 'day'], inplace=True)


train= train.drop('id',axis=1)


X= train.drop('y', axis=1)
y = train['y']
X_train_split , X_test_split, y_train_split , y_test_split = train_test_split(X,y, random_state=2, test_size=0.2 , stratify=y)


'''num = X.select_dtypes(include=['number']).columns.tolist()
cat = X.select_dtypes(include=['object']).columns.tolist()
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())])
preprocessing = ColumnTransformer([('pipe1', cat_pipeline, cat),
                                  ('pipe2', num_pipeline, num)])
models =[]
lgb_params = {
    "random_state": 42,
    "verbosity": -1,
    "n_estimators": 40000,
    "learning_rate": 0.0358306214515723,
    "min_child_samples": 83,
    "subsample": 0.8700304020753131,
    "colsample_bytree": 0.6169349166144594,
    "num_leaves": 228,
    "max_depth": 6,
    "max_bin": 3600,
    "reg_alpha": 3.700714656885025,
    "reg_lambda": 4.709578317972932,
}

oof_pred = np.zeros(len(X_train_split))
test_pred = np.zeros(len(test))
skf= StratifiedKFold(n_splits= 5, shuffle = True, random_state=42)
for fold, (train_idx, val_idx) in enumerate (skf.split(X_train_split, y_train_split)):
    X_train , X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
    y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]

    X_train_pre = preprocessing.fit_transform(X_train , y_train)
    X_val_pre = preprocessing.transform(X_val)
    X_test_trans_ = preprocessing.transform(test)

    
    dtrain = lgb.Dataset(X_train_pre , label= y_train)
    dval = lgb.Dataset(X_val_pre, label= y_val)
    
    model = lgb.train(lgb_params, dtrain, valid_sets=[dtrain , dval])
    y_pred = model.predict(X_val_pre)

    oof_pred[val_idx] = y_pred
    test_pred += model.predict(X_test_trans_)/skf.n_splits
    models.append(model)
joblib.dump(models, 'models.pkl')
overall_auc = roc_auc_score(y_train_split, oof_pred)
print(f"\nOverall Cross-Validation AUC: {overall_auc:.5f}")


X_test_full_pre = preprocessing.fit_transform(X)
X_test_trans = preprocessing.transform(test)

y_pred_test = np.mean([m.predict(X_test_trans) for m in models], axis=0)'''


'''cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessing = ColumnTransformer([
    ('pipe1', cat_pipeline, cat),
    ('pipe2', num_pipeline, num)
])

xgb_params = {
        'n_estimators': 8000,         
        'max_leaves': 127,            
        'min_child_weight': 1.5,     
        'max_depth': 0,               
        'grow_policy': 'lossguide',   
        'learning_rate': 0.008,      
        'tree_method': 'hist',        
        'subsample': 0.85,            
        'colsample_bylevel': 0.7,     
        'colsample_bytree': 0.75,       
        'colsample_bynode': 0.85,     
        'reg_alpha': 2.5,             
        'reg_lambda': 0.8,            
        'enable_categorical': True,    
        'max_cat_to_onehot': 1,       
        'device': 'cuda',            
        'n_jobs': -1,                 
        'random_state': 42,     
        'verbosity': 0,               
        'objective': 'binary:logistic',
    }


models_xgb = []
oof_pred_xgb = np.zeros(len(X_train_split))
test_pred = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_split, y_train_split)):
    X_train, X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
    y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]

    preprocessing.fit(X_train, y_train)
    X_train_pre = preprocessing.transform(X_train)
    X_val_pre = preprocessing.transform(X_val)
    X_test_trans_ = preprocessing.transform(test)

    dtrain = xgb.DMatrix(X_train_pre, label=y_train)
    dval = xgb.DMatrix(X_val_pre, label=y_val)
    dtest = xgb.DMatrix(X_test_trans)

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        verbose_eval=False
    )

    y_pred = model.predict(dval)
    oof_pred_xgb[val_idx] = y_pred
    test_pred_xgb += model.predict(dtest) / skf.n_splits
    models_xgb.append(model)

joblib.dump(models_xgb, 'xgb_models.pkl')

preprocessing.fit(X_train_split, y_train_split)
X_test_trans_final = preprocessing.transform(test)
dtest_final = xgb.DMatrix(X_test_trans_final)
y_pred_test_xgb = np.mean([m.predict(dtest_final) for m in models_xgb], axis=0)'''


'''oof_lbg = (oof_pred>=0.5).astype(int)
oof_xgb = (oof_pred_xgb >= 0.5).astype(int)
test_pred_lbg_label = (test_pred>=0.5).astype(int)
test_pred_xgb_label = (test_pred_xgb >= 0.5).astype(int)
print(classification_report(y_test_split, test_pred_lbg_label))
print(classification_report(y_test_split, test_pred_xgb_label))
print(confusion_matrix(y_test_split, test_pred_lbg_label))
print(confusion_matrix(y_test_split, test_pred_xgb_label))'''


num = X.select_dtypes(include=['number']).columns.tolist()
cat = X.select_dtypes(include=['object']).columns.tolist()
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessing = ColumnTransformer([
    ('pipe1', cat_pipeline, cat),
    ('pipe2', num_pipeline, num)
])

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 8000,
    'learning_rate': 0.008,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 8000,
    'max_leaves': 127,
    'min_child_weight': 1.5,
    'max_depth': 0,
    'grow_policy': 'lossguide',
    'learning_rate': 0.008,
    'tree_method': 'hist',
    'subsample': 0.85,
    'colsample_bylevel': 0.7,
    'colsample_bytree': 0.75,
    'colsample_bynode': 0.85,
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'enable_categorical': True,
    'max_cat_to_onehot': 1,
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'objective': 'binary:logistic'
}

models_lgb = []
oof_pred_lgb = np.zeros(len(X_train_split))
test_pred_lgb = np.zeros(len(test))

models_xgb = []
oof_pred_xgb = np.zeros(len(X_train_split))
test_pred_xgb = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_split, y_train_split)):
    X_train, X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
    y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]

    preprocessing.fit(X_train, y_train)
    X_train_pre = preprocessing.transform(X_train)
    X_val_pre = preprocessing.transform(X_val)
    X_test_pre = preprocessing.transform(test)

    dtrain_lgb = lgb.Dataset(X_train_pre, label=y_train)
    dval_lgb = lgb.Dataset(X_val_pre, label=y_val, reference=dtrain_lgb)

    model_lgb = lgb.train(
        params=lgb_params,
        train_set=dtrain_lgb,
        valid_sets=[dtrain_lgb, dval_lgb]
    )

    y_pred_lgb = model_lgb.predict(X_val_pre)
    oof_pred_lgb[val_idx] = y_pred_lgb
    test_pred_lgb += model_lgb.predict(X_test_pre) / skf.n_splits
    models_lgb.append(model_lgb)

    dtrain_xgb = xgb.DMatrix(X_train_pre, label=y_train)
    dval_xgb = xgb.DMatrix(X_val_pre, label=y_val)
    dtest_xgb = xgb.DMatrix(X_test_pre)

    model_xgb = xgb.train(
        params=xgb_params,
        dtrain=dtrain_xgb,
        evals=[(dtrain_xgb, 'train'), (dval_xgb, 'valid')]
    )

    y_pred_xgb = model_xgb.predict(dval_xgb)
    oof_pred_xgb[val_idx] = y_pred_xgb
    test_pred_xgb += model_xgb.predict(dtest_xgb) / skf.n_splits
    models_xgb.append(model_xgb)

joblib.dump(models_lgb, 'lgb_models.pkl')
joblib.dump(models_xgb, 'xgb_models.pkl')

y_pred_test_lgb = test_pred_lgb
y_pred_test_xgb = test_pred_xgb


weight_lgb = 0.5
weight_xgb = 0.5

y_pred_ensemble = (weight_lgb * y_pred_test_lgb) + (weight_xgb * y_pred_test_xgb)

submission = pd.DataFrame({
    'id': test['id'],
    'y': y_pred_ensemble
})

submission.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")


