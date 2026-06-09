import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import warnings, gc
warnings.filterwarnings('ignore')
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb  
from IPython.display import display
pd.set_option('display.max_columns', None)
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

!rm -rf /kaggle/working/*


train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')

print("Train Data Details..................\n")
print("#"*130)
print(f"Train Data Shape: {train.shape}")
print("#"*130)
print(f"Train Data Info: {train.info()}")
print("#"*130)
print(f"Check Train Data NUll Values: {train.isnull().sum()}")
print("#"*130)
print(f"Train data description: {train.describe()}")
print("#"*130)
display(train.head())


print("Test Data Details..................\n\n")
print("#"*130)
print(f"Test Data Shape: {test.shape}")
print("#"*130)
print(f"Test Data Info: {test.info()}")
print("#"*130)
print(f"Check Test Data NUll Values: {test.isnull().sum()}")
print("#"*130)
print(f"Test data description: {test.describe()}")
print("#"*130)
display(test.head())


train.drop(columns=["id"],axis=1,inplace=True)

def clean_data(df):
    df=df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Annual Income'].fillna(df['Annual Income'].mean(), inplace=True)
    df['Marital Status'].fillna(df['Marital Status'].mode()[0], inplace=True)
    df['Number of Dependents'].fillna(df['Number of Dependents'].median(), inplace=True)
    df['Occupation'].fillna(df['Occupation'].mode()[0], inplace=True)
    df['Health Score'].fillna(df['Health Score'].mean(), inplace=True)
    df['Previous Claims'].fillna(df['Previous Claims'].mean(), inplace=True)
    df['Vehicle Age'].fillna(df['Vehicle Age'].mode()[0], inplace=True)
    df['Credit Score'].fillna(df['Credit Score'].mean(), inplace=True)
    df['Insurance Duration'].fillna(df['Insurance Duration'].median(), inplace=True)
    df['Customer Feedback'].fillna(df['Customer Feedback'].mode()[0], inplace=True)
    return df

train=clean_data(train)
test=clean_data(test)


def ultimate_feature_engineering(df, is_train=True):
    df = df.copy()
    
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['year']       = df['Policy Start Date'].dt.year
    df['month']      = df['Policy Start Date'].dt.month
    df['day']        = df['Policy Start Date'].dt.day
    df['dow']        = df['Policy Start Date'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    df['log_income']            = np.log1p(df['Annual Income'])
    df['income_per_age']        = df['Annual Income'] / (df['Age'] + 1)
    df['income_per_dependent'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
    df['high_income']           = (df['Annual Income'] > 100000).astype(int)
    
    df['age_group']      = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=[0,1,2,3]).astype(int)
    df['age_x_income']   = df['Age'] * df['log_income']
    df['age_x_dependents'] = df['Age'] * df['Number of Dependents']
    
    df['smoker']        = (df['Smoking Status'] == 'Yes').astype(int)
    df['no_exercise']   = (df['Exercise Frequency'] == 'Never').astype(int)
    df['risk_score']    = df['smoker']*3 + df['no_exercise']*2 + (df['Health Score'] < 20).astype(int)*2
    
    df['credit_group']   = pd.qcut(df['Credit Score'], q=10, labels=False, duplicates='drop')
    df['new_car']        = (df['Vehicle Age'] <= 3).astype(int)
    df['long_duration']  = (df['Insurance Duration'] >= 5).astype(int)
    df['many_claims']    = (df['Previous Claims'] >= 3).astype(int)
    df['premium_policy'] = (df['Policy Type'] == 'Premium').astype(int)
    
    df['income_x_risk']     = df['log_income'] * df['risk_score']
    df['age_x_risk']        = df['Age'] * df['risk_score']
    df['credit_x_income']   = df['Credit Score'] * df['log_income']
    df['health_x_income']   = df['Health Score'] * df['log_income']
    
    freq_cols = ['Gender','Marital Status','Education Level','Occupation','Location',
                 'Policy Type','Property Type','Smoking Status','Exercise Frequency']
    for col in freq_cols:
        df[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))
    
    df = df.drop(columns=['Policy Start Date','Customer Feedback'], errors='ignore')
    
    return df


train=ultimate_feature_engineering(train,is_train=True)
test=ultimate_feature_engineering(test,is_train=False)


def fit_label_encoders(train_df):
    le_dict = {}
    for col in tqdm(train_df.columns, desc="Encoding train columns"):
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            le.fit(train_df[col])
            le_dict[col] = le
            train_df[col] = le.transform(train_df[col])
    return train_df, le_dict

def transform_with_encoders(df, le_dict):
    df_encoded = df.copy()
    for col in tqdm(le_dict.keys(), desc="Encoding test columns"):
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = df_encoded[col].map(lambda s: le_dict[col].transform([s])[0] if s in le_dict[col].classes_ else -1)
    return df_encoded

train, encoders = fit_label_encoders(train)
test = transform_with_encoders(test, encoders)



target = train['Premium Amount']
train = train.drop(['Premium Amount'], axis=1)
test = test.drop(columns=['id'], axis=1)


# 5-fold only
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_pred  = np.zeros(len(train))
test_pred = np.zeros(len(test))

print("Starting 5-fold CatBoost (log-target) — FAST & STRONG\n")
start_total = time.time()

for fold, (ti, vi) in enumerate(kf.split(train)):
    fold_start = time.time()
    print(f"Fold {fold+1}/5 training... ", end="", flush=True)
    
    X_tr, X_va = train.iloc[ti], train.iloc[vi]
    y_tr = np.log1p(target.iloc[ti])
    y_va = np.log1p(target.iloc[vi])

    model = cb.CatBoostRegressor(
        iterations            = 20000,
        learning_rate         = 0.05,
        depth                 = 10,
        l2_leaf_reg           = 3,
        random_strength       = 0.8,
        bagging_temperature   = 0.2,
        border_count          = 254,
        task_type             = 'GPU',
        devices               = '0',
        random_state          = 42,
        verbose               = False,
        early_stopping_rounds = 300,
        use_best_model        = True
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    oof_pred[vi] = np.expm1(model.predict(X_va))
    test_pred += np.expm1(model.predict(test)) / 5   # 5 folds

    fold_time = time.time() - fold_start
    folds_left = 4 - fold
    est_remaining = folds_left * fold_time

    rmsle = mean_squared_log_error(target.iloc[vi], oof_pred[vi]) ** 0.5
    print(f"Done | RMSLE: {rmsle:.6f} | Time: {fold_time/60:.1f}min | "f"{folds_left} left ≈ {est_remaining/60:.1f}min")

# Final
total_time = time.time() - start_total
cv_score = mean_squared_log_error(target, oof_pred) ** 0.5

print(f"\nAll 5 folds done in {total_time/60:.1f} minutes")
print(f"FINAL 5-fold CV RMSLE: {cv_score:.6f} ← Your expected LB score")


sub = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')
sub['Premium Amount'] = test_pred
sub.to_csv('submission.csv', index=False)
print("submission.csv saved")




