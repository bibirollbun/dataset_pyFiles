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


# load essential model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration
RANDOM_SEED = 42
N_FOLDS = 10
USE_GPU = True
np.random.seed(RANDOM_SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv') 


print("______________Train data______________")
print(train.head())
print("______________Test data______________")
print(test.head())
print(sample_submission.head())


# basic statistics
print(train.describe())


train.isnull().sum()


# Create subplots for numeric data 
# annual_income, debt_to_income_ratio, credit_score, loan_amount, interest_rate
fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # 1 row, 5 columns

axs[0].hist(train['annual_income'], bins=20, color='blue', edgecolor='black')
axs[0].set_title('annual_income')

axs[1].hist(train['debt_to_income_ratio'], bins=20, color='blue', edgecolor='black')
axs[1].set_title('debt_to_income_ratio')

axs[2].hist(train['credit_score'], bins=20, color='blue', edgecolor='black')
axs[2].set_title('credit_score')

axs[3].hist(train['loan_amount'], bins=20, color='blue', edgecolor='black')
axs[3].set_title('loan_amount')

axs[4].hist(train['interest_rate'], bins=20, color='blue', edgecolor='black')
axs[4].set_title('interest_rate')
#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()


#gender marital_status education_level employment_status loan_purpose grade_subgrade
#the feature above need to be encoded 
# we use "Label Encodeing" because its simple to set up

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
gender_encoded = encoder.fit_transform(train["gender"])
marital_status_encoded = encoder.fit_transform(train["marital_status"])
education_level_encoded = encoder.fit_transform(train["education_level"])
employment_status_encoded = encoder.fit_transform(train["employment_status"])
loan_purpose_encoded = encoder.fit_transform(train["loan_purpose"])
grade_subgrade_encoded = encoder.fit_transform(train["grade_subgrade"])


# Create subplots
fig, axs = plt.subplots(1, 6, figsize=(10, 4))  # 1 row, 4 columns
axs[0].hist(gender_encoded, color='orange', edgecolor='black')
axs[0].set_title('gender')
axs[1].hist(marital_status_encoded, color='orange', edgecolor='black')
axs[1].set_title('marital_status')
axs[2].hist(education_level_encoded, color='orange', edgecolor='black')
axs[2].set_title('education_level')
axs[3].hist(employment_status_encoded,color='orange', edgecolor='black')
axs[3].set_title('employment_status')
axs[4].hist(loan_purpose_encoded, color='orange', edgecolor='black')
axs[4].set_title('loan_purpose')
axs[5].hist(grade_subgrade_encoded, color='orange', edgecolor='black')
axs[5].set_title('grade_subgrade')
#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()


from sklearn.preprocessing import PolynomialFeatures
def preprocess_dataframe(df: pd.DataFrame, drop_cols: list = None) -> pd.DataFrame:
    df = df.copy()

    # Handle missing values
    # Althought there are no missing data there might have missing data in future input data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Step 0: Drop specified columns 
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')  # 'ignore' avoids errors if column not found

    # Step 1: Encode boolean columns
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Step 2: Label encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Step 3: relate feature 
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1e-6)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    
    return df


preprocess_train = preprocess_dataframe(train,drop_cols=['loan_paid_back','id'])


preprocess_train.head()


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna


#1. preprocess data for model
train_data = preprocess_dataframe(train,drop_cols=['id'])
test_ids = test['id']
test_data = preprocess_dataframe(test,drop_cols=['id'])

train_x = train_data.drop(['loan_paid_back'] , axis=1)
train_y = train_data['loan_paid_back']


# 1. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    train_x, train_y, test_size=0.3, random_state=RANDOM_SEED
)

# 2. 定義目標函數
def objective(trial):
    # 定義要搜尋的超參數範圍
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": RANDOM_SEED,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
    }

    # 建立模型
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)

    # 預測並計算 AUC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc

# 3. 建立並執行 Optuna study
study = optuna.create_study(direction="maximize")  # 最大化 AUC
study.optimize(objective, n_trials=50)  # 跑 50 次試驗

# 4. 最佳結果
print("最佳參數:", study.best_params)
print("最佳 AUC:", study.best_value)




# 假設 Optuna 找到的最佳參數存在 study.best_params
best_params = study.best_params

# 1. 建立最佳模型
best_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    random_state=RANDOM_SEED
)

# 2. 用訓練集重新訓練模型
best_model.fit(X_train, y_train)

# 3. 在測試集上產生預測機率
y_pred_prob = best_model.predict_proba(test_data)[:, 1]

# 4. Submission file prepare

print("Creating submission file")
submission = pd.DataFrame({'loan_paid_back': y_pred_prob}, index=test_ids)
submission.index.name = 'id'

print("Saving submission file")
submission.to_csv('submission.csv', header=True)

print(f"Submission file created: {submission.shape}")
print("First 5 rows of submission:")
print(submission.head())




