


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install catboost
!pip install -U scikit-learn imbalanced-learn
!pip install optuna




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import xgboost as xgb
import catboost
import optuna




train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df.head(20)



train_df.info()


def unique(df):
  for col in df.columns:
    print(f"value counts for column:{col}")
    print(df[col].value_counts())
    print("-"*100)
unique(train_df)


train_df.replace('unknown',np.nan,inplace=True)
test_df.replace('unknown',np.nan,inplace=True)
train_df.info()


categorical_df_train = train_df.select_dtypes(include=['object'])
numerical_df_train = train_df.select_dtypes(exclude=['object'])
categorical_df_test = test_df.select_dtypes(include=['object'])
numerical_df_test = test_df.select_dtypes(exclude=['object'])
numerical_df_train.info()


cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')
categorical_df_train = pd.DataFrame(cat_imputer.fit_transform(categorical_df_train),columns=categorical_df_train.columns)
categorical_df_test = pd.DataFrame(cat_imputer.transform(categorical_df_test),columns=categorical_df_test.columns)


categorical_df_train.head(20)


eligible = ['default', 'housing', 'loan']

for col in eligible:
    for df in [categorical_df_train, categorical_df_test]:
        df[col] = df[col].replace({'yes': 1, 'no': 0, 'unknown': -1})

numerical_df_train = pd.concat([numerical_df_train, categorical_df_train[eligible]], axis=1)
numerical_df_test = pd.concat([numerical_df_test, categorical_df_test[eligible]], axis=1)

categorical_df_train.drop(eligible, axis=1, inplace=True)
categorical_df_test.drop(eligible, axis=1, inplace=True)



def corr(df,method='pearson'):
  corr = df.corr(method=method)
  plt.figure(figsize=(20,16))
  sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
  plt.title(f"{method.capitalize()} Correlation Matrix")
  plt.tight_layout()
  plt.show()
  return corr
print(corr(numerical_df_train.drop(columns=['y'])))


combining_features = ['loan','housing','default','balance','campaign']
set_features = set()
for i in combining_features:
  for j in combining_features:
    if i!= j and ((i,j) not in set_features) and(j,i) not in set_features:
      set_features.add((i,j))
      numerical_df_train[i + '+' + j] = numerical_df_train[i] + numerical_df_train[j]
      numerical_df_test[i + '+' + j] = numerical_df_test[i] + numerical_df_test[j]
      numerical_df_train[i + '*' + j] = numerical_df_train[i] * numerical_df_train[j]
      numerical_df_test[i + '*' + j] = numerical_df_test[i] * numerical_df_test[j]
      numerical_df_train[i + '-' + j] = numerical_df_train[i] - numerical_df_train[j]
      numerical_df_test[i + '-' + j] = numerical_df_test[i] - numerical_df_test[j]


corr(numerical_df_train)


numerical_df_train.drop(columns=['y','id'],inplace=True)
numerical_df_test.drop(columns=['id'],inplace=True)



standard = StandardScaler()
numerical_df_train = pd.DataFrame(standard.fit_transform(numerical_df_train),columns=numerical_df_train.columns)
numerical_df_test = pd.DataFrame(standard.transform(numerical_df_test),columns=numerical_df_test.columns)
X = pd.concat([numerical_df_train,categorical_df_train],axis=1)
X_t = pd.concat([numerical_df_test,categorical_df_test],axis=1)
y = train_df['y']


def add_features_numeric_only(df):
    df = df.copy()

    # Age features
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_young_adult'] = df['age'].between(18, 30).astype(int)
    df['age_decade'] = (df['age'] // 10) * 10
    df['age_zscore'] = (df['age'] - df['age'].mean()) / df['age'].std()
    df['age_bin'] = pd.cut(df['age'], bins=range(15, 100, 5), labels=False)

    # Label encode categorical columns safely
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    # Job features
    job_freq = df['job'].value_counts(normalize=True)
    df['job_freq'] = df['job'].map(job_freq).fillna(0)
    df['job_is_high_profile'] = df['job'].isin(['management', 'admin.', 'technician']).astype(int)
    df['is_self_employed'] = df['job'].isin(['self-employed', 'entrepreneur']).astype(int)

    # Marital features
    df['is_married'] = (df['marital'] == 'married').astype(int)
    df['is_single_or_divorced'] = df['marital'].isin(['single', 'divorced']).astype(int)

    # Education features
    df['is_educated'] = df['education'].isin(['tertiary', 'secondary']).astype(int)
    df['unknown_education'] = (df['education'] == 'unknown').astype(int)
    df['edu_job_match'] = ((df['education'] == 'tertiary') & df['job_is_high_profile'].astype(bool)).astype(int)

    # Financial features
    df['balance_log'] = df['balance'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    df['is_balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_zscore'] = (df['balance'] - df['balance'].mean()) / df['balance'].std()
    df['balance_bucket'] = pd.qcut(df['balance'], 5, labels=False, duplicates='drop')
    df['has_high_balance'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)

    # Loan & Housing features
    df['has_any_loan'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
    df['has_both_loans'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)

    # Loan balance ratio: balance / loan indicator (replace 0 with NaN to avoid div by zero)
    loan_indicator = (df['loan'] == 'yes').astype(int).replace(0, np.nan)
    df['loan_balance_ratio'] = df['balance'] / loan_indicator

    # Contact features
    df['is_mobile_contact'] = (df['contact'] == 'cellular').astype(int)
    df['unknown_contact'] = (df['contact'] == 'unknown').astype(int)
    df['preferred_contact_score'] = df['contact'].map({'cellular': 2, 'telephone': 1, 'unknown': 0}).fillna(0)

    # Campaign & Previous Contact features
    df['calls_per_day'] = df['campaign'] / df['day'].replace(0, np.nan)
    df['multiple_contacts_flag'] = (df['campaign'] > 3).astype(int)
    df['pdays_flag'] = (df['pdays'] != -1).astype(int)
    df['days_since_last_contact_bucket'] = pd.cut(df['pdays'], bins=[-2,0,30,90,180,999], labels=False)
    df['previous_contact_ratio'] = df['previous'] / df['campaign'].replace(0, np.nan)

    # Temporal features
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df['month_enc'] = df['month'].map(month_map).fillna(0).astype(int)
    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}  # winter=0,spring=1,summer=2,fall=3
    df['season'] = df['month_enc'].map(season_map).fillna(-1).astype(int)
    df['is_month_end'] = (df['day'] > 25).astype(int)
    df['day_of_week_estimate'] = df['day'] % 7
    df['is_q2'] = df['month'].isin(['apr','may','jun']).astype(int)

    # Interaction features (example 10)
    df['age_x_balance'] = df['age'] * df['balance']
    df['campaign_x_previous'] = df['campaign'] * df['previous']
    df['pdays_x_poutcome'] = df['pdays'] * df['poutcome_enc'].fillna(0)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'].replace(0, np.nan))
    df['duration_per_campaign'] = df['duration'] / (df['campaign'].replace(0, np.nan))
    df['age_bin_x_is_married'] = df['age_bin'].fillna(-1).astype(int) * df['is_married']
    df['job_enc_x_loan'] = df['job_enc'] * (df['loan'] == 'yes').astype(int)
    df['balance_log_x_is_balance_pos'] = df['balance_log'] * df['is_balance_positive']
    df['calls_per_day_x_multiple_contacts'] = df['calls_per_day'] * df['multiple_contacts_flag']
    df['season_x_is_q2'] = df['season'] * df['is_q2']

    # Drop original categorical string columns to avoid dtype issues
    original_cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df = df.drop(columns=[c for c in original_cat_cols if c in df.columns])

    # Fill any remaining NaNs (for safety)
    df = df.fillna(0)

    # Ensure all columns are numeric type
    for c in df.columns:
        if df[c].dtype == 'bool':
            df[c] = df[c].astype(int)
        elif df[c].dtype.name == 'category':
            df[c] = df[c].astype(int)

    return df
X = add_features_numeric_only(X)
X_t = add_features_numeric_only(X_t)


smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


X_train,X_val,y_train,y_val = train_test_split(X,y,stratify=y,random_state=42)


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "error",  # 'error' = 1 - accuracy
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

        # GPU params
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "predictor": "gpu_predictor"
    }

    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return 1 - acc  
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective, n_trials=10)

print("Best params:", study_xgb.best_params)
print("Best score:", study_xgb.best_value)
best_xgb_model = xgb.XGBClassifier(**study_xgb.best_params, random_state=42, use_label_encoder=False)
best_xgb_model.fit(X_train, y_train)
preds = best_xgb_model.predict(X_val)
print("Final Accuracy:", accuracy_score(y_val, preds))


def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device_type": "gpu",              
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  

best_params_lgb = study.best_params
print("Best params for LightGBM:", best_params_lgb)


xgb_model = xgb.XGBClassifier(
    **study_xgb.best_params,
    objective="binary:logistic",
    tree_method="gpu_hist",       
    predictor="gpu_predictor",
    eval_metric="logloss",
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    **best_params_lgb,
    objective="binary",
    metric="binary_error",
    verbosity=-1,
    device_type="gpu",         
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=42
)

cat_model = CatBoostClassifier(
    task_type="GPU",              
    devices="0",
    verbose=0,
    random_state=42
)
voting_clf = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("cat", cat_model)
    ],
    voting="soft"
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X, y, cv=skf, scoring="accuracy")
print(f"Mean CV Accuracy (GPU): {np.mean(scores):.4f}")


voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_val)
y_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
from sklearn.metrics import accuracy_score, roc_auc_score
print("Accuracy:", accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))


def test_pipeline(X_t,model):
  test_preds = model.predict(X_t)
  submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
  submission['y'] = test_preds
  submission.to_csv('submission_bank1.csv', index=False)
  return submission
submision_pipe = test_pipeline(X_t,voting_clf)


submision_xgb.head(25)

