import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df_train = pd.read_csv("/kaggle/input/iriisss/train.csv")
df_test = pd.read_csv("/kaggle/input/iriisss/test.csv")
submission = pd.read_csv("/kaggle/input/iriisss/sample_submission.csv")


df_train.shape


df_train.head()


df_test.head()


submission.head()


df_train.isna().sum()


df_train.duplicated().sum()


df_train = df_train.drop("id", axis=1)


df_train.shape


df_train.duplicated().sum()


df_train.isna().sum()


df_train = df_train.drop_duplicates()


df_train.duplicated().sum()


df_train = df_train.dropna()


df_train.isna().sum()


df_train.shape


df_train.head()


df_train.head()


numeric_data = df_train.select_dtypes(include=[np.number])
categorical_data = df_train.select_dtypes(exclude=[np.number])


numeric_data.head()


categorical_data.head()


# categorical_data = categorical_data.drop("Surname", axis=1)
categorical_data.head()


corr = numeric_data.corr()
corr.style.background_gradient(cmap='coolwarm')


# df_train = df_train.drop("Surname", axis=1)

test_ids = df_test["id"]
# df_test = df_test.drop(["Surname", "id"], axis=1)
df_test = df_test.drop(["id"], axis=1)


df_test.head()


df_train.head()


df_train.head()


df_test.head()


#customer_state, customer_area_code, has_voice_mail_plan, voice_mail_message_count
df_train = df_train.drop(columns=['customer_state'])
df_test = df_test.drop(columns=['customer_state'])

def fill_voice_mail_data(df):
    df['voice_mail_message_count'] = df.apply(
        lambda row: 28.84 * row['has_voice_mail_plan'] if pd.isna(row['voice_mail_message_count']) else row['voice_mail_message_count'], axis=1
    )
    df['has_voice_mail_plan'] = df.apply(
        lambda row: 1 if pd.isna(row['has_voice_mail_plan']) and pd.notna(row['voice_mail_message_count']) and row['voice_mail_message_count'] > 0 else 0 
        if pd.isna(row['has_voice_mail_plan']) else row['has_voice_mail_plan'], axis=1
    )
    df['voice_mail_message_count'] = df['voice_mail_message_count'].fillna(0)
    df['has_voice_mail_plan'] = df['has_voice_mail_plan'].map({True: 1, False: 0})
    return df
    
df_train = fill_voice_mail_data(df_train)
df_test = fill_voice_mail_data(df_test)
df_train['customer_area_code'] = df_train['customer_area_code'].str.extract('(\d+)')
df_test['customer_area_code'] = df_test['customer_area_code'].str.extract('(\d+)')
df_train['customer_area_code'] = df_train['customer_area_code'].fillna(df_train['customer_area_code'].mode()[0])
df_test['customer_area_code'] = df_test['customer_area_code'].fillna(df_test['customer_area_code'].mode()[0])

#customer_account_duration
df_train['customer_account_duration'] = df_train['customer_account_duration'].fillna(df_train['customer_account_duration'].mean())
df_test['customer_account_duration'] = df_test['customer_account_duration'].fillna(df_test['customer_account_duration'].mean())

#has_international_plan
mapping = {'False': 0, 'false': 0, 'no': 0, 'No': 0, 'FALSE': 0, 'NO': 0, '0': 0,
           'True': 1, 'true': 1, 'yes': 1, 'Yes': 1, 'TRUE': 1, 'YES': 1, '1': 1}
df_train['has_international_plan'] = df_train['has_international_plan'].replace(mapping).astype('float64')
df_test['has_international_plan'] = df_test['has_international_plan'].replace(mapping).astype('float64')
df_train['has_international_plan'] = df_train['has_international_plan'].fillna(0) #df_train['churn'].map({0: 0, 1: 1})
df_test['has_international_plan'] = df_test['has_international_plan'].fillna(0)

#daytime, evening, nighttime, intl : total_minutes, total_charges
def fill_missing_values(df, minutes_col, charges_col, rate, base_charge=0):
    df[charges_col] = df.apply(
        lambda row: rate * row[minutes_col] + base_charge if pd.isna(row[charges_col]) else row[charges_col], axis=1
    )
    df[minutes_col] = df.apply(
        lambda row: (row[charges_col] - base_charge) / rate if pd.isna(row[minutes_col]) else row[minutes_col], axis=1
    )
    df[minutes_col] = df[minutes_col].fillna(df[minutes_col].median()).astype('float64')
    df[charges_col] = df[charges_col].fillna(df[charges_col].median()).astype('float64')
    
columns_info = [
    ('daytime_total_minutes', 'daytime_total_charges', 0.17, 0.30),
    ('evening_total_minutes', 'evening_total_charges', 0.08, 0.13),
    ('nighttime_total_minutes', 'nighttime_total_charges', 0.04, 0.08),
    ('intl_total_minutes', 'intl_total_charges', 0.27, 0)
]
for minutes_col, charges_col, rate, base_charge in columns_info:
    fill_missing_values(df_train, minutes_col, charges_col, rate, base_charge)
    fill_missing_values(df_test, minutes_col, charges_col, rate, base_charge)
    
#daytime, evening, nighttime, intl : total_calls
call_columns = ['daytime_total_calls','evening_total_calls','nighttime_total_calls','intl_total_calls']
for col in call_columns:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_test[col].median())

#customer_service_call_count
df_train['customer_service_call_count'] = df_train['customer_service_call_count'].fillna(0)
df_test['customer_service_call_count'] = df_test['customer_service_call_count'].fillna(0)

def feature_engineering(df):
    
    df['daytime_avg_minutes_per_call'] = df['daytime_total_minutes'] / (df['daytime_total_calls'] + 1e-6)
    df['evening_avg_minutes_per_call'] = df['evening_total_minutes'] / (df['evening_total_calls'] + 1e-6)
    df['nighttime_avg_minutes_per_call'] = df['nighttime_total_minutes'] / (df['nighttime_total_calls'] + 1e-6)
    df['intl_avg_minutes_per_call'] = df['intl_total_minutes'] / (df['intl_total_calls'] + 1e-6)
    df['total_minutes'] = (df['daytime_total_minutes'] +df['evening_total_minutes'] +df['nighttime_total_minutes'] +df['intl_total_minutes'])
    df['total_calls'] = (df['daytime_total_calls'] +df['evening_total_calls'] +df['nighttime_total_calls'] +df['intl_total_calls'])
    df['total_charges'] = (df['daytime_total_charges'] +df['evening_total_charges'] +df['nighttime_total_charges'] +df['intl_total_charges'])
    df['charges_per_minute'] = df['total_charges'] / (df['total_minutes'] + 1e-6)
    df['service_calls_per_duration'] = df['customer_service_call_count'] / (df['customer_account_duration'] + 1e-6)

    df['charges_to_account_duration'] = df['total_charges'] / (df['customer_account_duration'] + 1e-6)
    df['charges_per_service_call'] = df['total_charges'] / (df['customer_service_call_count'] + 1e-6)
    df['total_minutes_per_month'] = df['total_minutes'] / (df['customer_account_duration'] + 1e-6)
    df['total_calls_per_month'] = df['total_calls'] / (df['customer_account_duration'] + 1e-6)
    df['total_cost_efficiency'] = df['total_charges'] / (df['total_minutes'] + 1e-6)
    df['day_vs_night_usage'] = df['daytime_total_minutes'] / (df['nighttime_total_minutes'] + 1e-6)
    df['intl_vs_total_usage'] = df['intl_total_minutes'] / (df['total_minutes'] + 1e-6)
    df['service_calls_per_month'] = df['customer_service_call_count'] / (df['customer_account_duration'] + 1e-6)
    df['day_night_charge_ratio'] = df['daytime_total_charges'] / (df['nighttime_total_charges'] + 1e-6)
    df['voice_mail_to_total_calls'] = df['voice_mail_message_count'] / (df['total_calls'] + 1e-6)
    df['intl_calls_ratio'] = df['intl_total_calls'] / (df['total_calls'] + 1e-6)
    df['is_high_usage'] = (df['total_minutes'] > df['total_minutes'].mean()).astype(int)
    df['frequent_service_calls'] = (df['customer_service_call_count'] > df['customer_service_call_count'].mean()).astype(int)

    return df

#median_income = df_train['person_income'].median()
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


enc = LabelEncoder()

categorical_features = ["customer_area_code"]

for cat_feat in categorical_features:
    df_train[cat_feat] = enc.fit_transform(df_train[cat_feat])
    df_test[cat_feat] = enc.transform(df_test[cat_feat])


df_train.head()


df_train.describe()


# numeric_data      = df_train.drop(["Geography", "Gender", "Tenure", "HasCrCard", "IsActiveMember", "Geo_Gender", "IsSenior", "QualityOfBalance", "CreditScoreTier", "IsActive_by_CreditCard", "Products_Per_Tenure", "Customer_Status", "Exited"], axis=1)
# numeric_data_test = df_test.drop(["Geography", "Gender", "Tenure", "HasCrCard", "IsActiveMember", "Geo_Gender", "IsSenior", "QualityOfBalance", "CreditScoreTier", "IsActive_by_CreditCard", "Products_Per_Tenure", "Customer_Status"], axis=1)

# numeric_data      = numeric_data.drop("CustomerId", axis=1)
# numeric_data_test = numeric_data_test.drop("CustomerId", axis=1)
# numeric_data.head()


# numeric_data.head()


# numeric_data_test.head()


# scaler = StandardScaler()

# scaled_numerical_data_train = scaler.fit_transform(numeric_data)
# scaled_numerical_data_test  = scaler.transform(numeric_data_test)    


# scaled_numerical_data_train[0]


# pca_columns = [f'Surname_PCA_{i+1}' for i in range(10)]
# features = ["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary"]
# for i in pca_columns:
#     features.append(i)

# for i, feat in enumerate(features):
#     l = []
    
#     for j in range(len(scaled_numerical_data_train)):
#         l.append(scaled_numerical_data_train[j][i])
    
#     df_train[feat] = l
    
#     l = []
    
#     for j in range(len(scaled_numerical_data_test)):
#         l.append(scaled_numerical_data_test[j][i])
    
#     df_test[feat]  = l


df_train.head()


df_test.head()


X = df_train.drop("churn", axis=1)
y = df_train["churn"]
X_test = df_test


lgbm = LGBMClassifier(**{  'objective'           : 'binary',
                           'boosting_type'       : 'gbdt',
                           'metric'              : "auc",
                           'random_state'        : 42,
                           'colsample_bytree'    : 0.56,
                           'subsample'           : 0.35,
                           'learning_rate'       : 0.05,
                           'max_depth'           : 8,
                           'n_estimators'        : 1000,
                           'num_leaves'          : 140,
                           'reg_alpha'           : 0.14,
                           'reg_lambda'          : 0.85,
                           'verbosity'           : -1, 
                          })
xgb  = XGBClassifier(**{  'objective'             : 'binary:logistic',
                          'eval_metric'           : "auc",
                          'random_state'          : 42,
                          'colsample_bytree'      : 0.25,
                          'learning_rate'         : 0.07,
                          'max_depth'             : 8,
                          'n_estimators'          : 800,                         
                          'reg_alpha'             : 0.09,
                          'reg_lambda'            : 0.70,
                          'min_child_weight'      : 22,
                          'verbosity'             : 0,
                         })
cat  = CatBoostClassifier(**{
                         'iterations'            : 10000,
                         'objective'             : 'Logloss',
                         'eval_metric'           : "AUC",
                         'early_stopping_rounds' : 1000,
                         'bagging_temperature'   : 0.1,
                         'colsample_bylevel'     : 0.88,
                         'iterations'            : 1000,
                         'learning_rate'         : 0.065,
                         'max_depth'             : 7,
                         'l2_leaf_reg'           : 1,
                         'min_data_in_leaf'      : 25,
                         'random_strength'       : 0.1, 
                         'max_bin'               : 100,
                         'verbose'               : 0,
                        })

vote = VotingClassifier(estimators=[('lgbm', lgbm), ('xgb', xgb), ('cat', cat)], voting='soft', weights=[2, 1, 1])

# Initialize an empty array to hold the submission predictions
submission_predictions = []

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# save aucs
aucs = []
ind = 1

for train_index, test_index in kf.split(X, y):
    print(f"============== Working on fold #{ind} ================")
    X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[test_index]

    print()
    print("               Fitting the voting model...              ")
    # Fit the model
    vote.fit(X_train_kf, y_train_kf)

    print()
    print("            Predicting on the validation data           ")
    # Predict probabilities for validation set
    y_pred_val = vote.predict_proba(X_val_kf)[:, 1]

    # Calculate AUC for validation set
    auc_val = roc_auc_score(y_val_kf, y_pred_val)
    print()
    print(f"           Validation ROC AUC Score: {auc_val}        ")
    
    aucs.append(auc_val)

    print()
    print("             Predicting on submission data...")
    # Predict probabilities for test set (df_test)
    y_pred_test = vote.predict_proba(X_test)[:, 1]
    submission_predictions.append(y_pred_test)
    
    print()
    print(f"                 Fold #{ind} finished !                ")
    
    ind+=1


print(f"Average ROC AUC Score: {sum(aucs) / len(aucs)}")

# Average predictions from different folds
avg_submission = pd.DataFrame(submission_predictions).mean(axis=0)

submission["churn"] = avg_submission

# Save submission to CSV
submission.to_csv("submission.csv", index=False)

submission.head()

