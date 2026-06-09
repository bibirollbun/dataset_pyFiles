import numpy as np
import pandas as pd 
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import optuna
import warnings

warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/train.csv')
df_test = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')


# df_train.head()


# df_train.describe()


df_train.info()


col_drop = ['name', 'host_id', 'host_name', 'neighbourhood', 'neighbourhood_cleansed', 'host_location','host_neighbourhood']

df_train.drop(columns=col_drop, inplace=True)
df_test.drop(columns=col_drop, inplace=True)
df_train = df_train.drop('id', axis=1)

num_cols = df_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()


# plt.figure(figsize=(16, 10))
# sns.heatmap(df_train[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Heatmap Korelasi Antar Variabel Kuantitatif')
# plt.show()


freqs = df_train["city"].value_counts()
print(freqs)


freqs = df_train["property_type"].value_counts()
print(freqs)


freqs = df_train["room_type"].value_counts()
print(freqs)


freqs = df_train["city"].value_counts()
print(freqs)


df_train.duplicated().sum()


df_train = df_train.drop_duplicates()


#parse amenities
import ast

df_train['amenities_list'] = df_train['amenities'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df_test['amenities_list'] = df_test['amenities'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])    
from collections import Counter

all_amenities = Counter()
for row in df_train['amenities_list']:
    all_amenities.update(row)
    # print(f"unique amenities: {len(all_amenities)}")
    
top_amenities = [item for item, count in all_amenities.most_common(70)]


for df in [df_train, df_test]:
    df['num_amenities'] = df['amenities_list'].apply(len)
    for amenity in top_amenities:
        col_name = f'amenity_{amenity.lower().replace(" ", "_").replace("-", "_").replace("/", "_")}'
        df[col_name] = df['amenities_list'].apply(lambda x: amenity in x)
    df.drop(['amenities', 'amenities_list'], axis=1, inplace=True)
    
# for amenity in top_amenities:
#     df[f'amenity_{amenity.lower().replace(" ", "_").replace("-", "_").replace("/", "_")}'] = df['amenities_list'].apply(lambda x: amenity in x)
# df = df.drop('amenities', axis=1)
# df = df.drop('amenities_list', axis=1)


# df_train = amenities(df_train)

# df_train.head()


def binary(df):
    df['host_is_superhost'] = df['host_is_superhost'].map({'f': 0, 't': 1})
    df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'f': 0, 't': 1})
    df['host_identity_verified'] = df['host_identity_verified'].map({'f': 0, 't': 1})
    df['has_availability'] = df['has_availability'].map({'f': 0, 't': 1})
    return df

def property_transform(df):
    df['property_structure'] = df['property_type'].str.extract(r'in (.+)$')
    df['property_structure'] = df['property_structure'].fillna(
        df['property_type'].str.replace(r'^(Entire|Private|Shared) ', '', regex=True)
    )
    df = df.drop('property_type', axis=1)

    threshold = 350
    freqs = df["property_structure"].value_counts()
    
    common_structures = freqs[freqs >= threshold].index.tolist()
    
    df["property_structure"] = df["property_structure"].apply(
        lambda x: x if x in common_structures else "other"
    )
    return df

def encoding(df):
    cat_col = ['property_structure', 'room_type','host_response_time']

    encode = {}

    for col in cat_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df['lat_lon'] = df['latitude'] * df['longitude']
    df = df.drop('city', axis=1)
        # encode[col] = le  # save for test set

    # df[cat_col].head()

    # city_freq = df['city'].value_counts(normalize=True)
    # df['city_freq'] = df['city'].map(city_freq)
    # df = df.drop('city', axis=1)
    return df

def verifications(df):
    verif_cols = ['phone', 'email', 'work_email', 'photographer']
    
    df['host_verifications'] = df['host_verifications'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    for col in verif_cols:
        df[f'verif_{col}'] = df['host_verifications'].apply(lambda x: col in x).astype(bool)

    df = df.drop('host_verifications', axis=1)
    # df['phone'] = df['host_verifications'].str.contains('phone')
    # df['email'] = df['host_verifications'].str.contains('email')
    # df['work_email'] = df['host_verifications'].str.contains('work_email')
    # df['photographer'] = df['host_verifications'].str.contains('photographer')
    return df

def percentage_conv(df):
    cols_to_convert = ['host_response_rate', 'host_acceptance_rate']
    
    for col in cols_to_convert:
        df[col] = df[col].str.rstrip('%')
        df[col] = df[col].replace('', np.nan).astype(float) / 100
    return df

def date_convert(df):
    date_cols = ['host_since', 'first_review', 'last_review']
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    today = pd.to_datetime("2025-04-01")

    #age
    df['host_age_days'] = (today - df['host_since']).dt.days

    #recency
    df['last_activity'] = (today - df['last_review']).dt.days

    #first review
    df['first_activity'] = (today - df['first_review']).dt.days

    for cols in date_cols:
        df.drop(columns=[cols], inplace=True)
    return df


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def text_transform(df):
    transform_col = ['description', 'neighborhood_overview']
    n_components = 3

    for col in transform_col:
        df[col] = df[col].fillna("")  # Handle NaN
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df[col])

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_svd = svd.fit_transform(tfidf_matrix)

        for i in range(n_components):
            df[f'{col}_svd_{i}'] = tfidf_svd[:, i]

        df.drop(columns=[col], inplace=True)

    drop_col = ['host_about','bathrooms_text' ]
    
    for col in drop_col:
        df.drop(columns=[col], inplace=True)

    return df


df_train = text_transform(df_train)
df_train = binary(df_train)
df_train = percentage_conv(df_train)
df_train = verifications(df_train)
df_train = property_transform(df_train)

df_test = text_transform(df_test)
df_test = binary(df_test)
df_test = percentage_conv(df_test)
df_test = verifications(df_test)
df_test = property_transform(df_test)


pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')


freqs = df_train["property_structure"].value_counts()
print(freqs)


for col in df_train.columns:
    print(f"{col}: {df_train[col].dtype}")


df_train = encoding(df_train)
df_train = date_convert(df_train)

df_test = encoding(df_test)
df_test = date_convert(df_test)


for col in df_train.columns:
    print(f"{col}: {df_train[col].dtype}")

# verifs = ['phone', 'email', 'work_email', 'photographer']
# df_train[verifs].head()



df_test.describe()


# text_svd_cols = [col for col in df_train.columns if any(prefix in col for prefix in [
#     'description_svd_', 
#     'neighborhood_overview_svd_', 
#     'host_about_svd_', 
#     'bathrooms_text_svd_'
# ])]
# df_train_no_text = df_train.drop(columns=text_svd_cols)
# df_test_no_text = df_test.drop(columns=text_svd_cols)


# df_train_no_text.describe()


# y = df_train['price']
# X = df_train.drop('price', axis=1)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

# y = df_train_no_text['price']
# X = df_train_no_text.drop('price', axis=1)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

X_train = df_train.drop(columns=["price"])
y_train = df_train["price"]


# def objective(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'n_estimators': 2000,   # always set max trees
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha' : trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
#         'random_state': 42,
#         'n_jobs': -1
#     }
#     #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     rmse_scores = []

#     for train_idx, val_idx in kf.split(X_train):
#         X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model = xgb.XGBRegressor(**params)
#         model.fit(X_t, y_t,eval_set=[(X_v, y_v)],early_stopping_rounds=50,verbose=False)
#         preds = model.predict(X_v)
#         rmse = mean_squared_error(y_v, preds, squared=False)
#         rmse_scores.append(rmse)

#     mean_rmse = np.mean(rmse_scores)

#     print(f"\nTrial {trial.number}: Mean RMSE={mean_rmse:.5f}")
#     print("Params:", params)

#     return mean_rmse
    
# print("Starting Optuna for XGB...")
# study = optuna.create_study(direction='minimize')
# # study.enqueue_trial(**params)
# study.optimize(objective, n_trials=25)
# print("Finished. \n")
# print("Best parameters: ", study.best_params)
# print("Best RMSE: ", study.best_value)


# def objective_lgb(trial):
#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'boosting_type': 'gbdt',
#         'n_estimators': 2000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
#         'random_state': 42,
#         'verbosity': -1
#     }
    
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     rmse_scores = []

#     for train_idx, val_idx in kf.split(X_train):
#         X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         model = lgb.LGBMRegressor(**params)
#         model.fit(X_t, y_t,eval_set=[(X_v, y_v)],eval_metric="rmse", callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)])
#         preds = model.predict(X_v)
#         rmse = mean_squared_error(y_v, preds, squared=False)
#         rmse_scores.append(rmse)

#     mean_rmse = np.mean(rmse_scores)
#     print(f"Trial {trial.number} - Avg RMSE: {mean_rmse:.5f}")
#     return mean_rmse

# print("Starting Optuna for LightGBM...")
# study_lgb = optuna.create_study(direction='minimize')
# study_lgb.optimize(objective_lgb, n_trials=25)
# print("Finished. \n")
# print("Best parameters for LGB: ", study_lgb.best_params)
# print("Best RMSE for LGB: ", study_lgb.best_value)


#From failed run on version 2
#XGB
# Best parameters:  {'learning_rate': 0.02730109828631519, 'max_depth': 9, 'subsample': 0.854847064173804, 'colsample_bytree': 0.6421317014498971, 'reg_alpha': 0.07672242008556412, 'reg_lambda': 0.027970833737134406}
# Best RMSE:  89.20407531978333

#LGB
#Best parameters for LGB:  {'learning_rate': 0.025685142866834026, 'num_leaves': 85, 'feature_fraction': 0.9013420574831337, 'bagging_fraction': 0.7592214624078952, 'bagging_freq': 3, 'lambda_l1': 1.976024281017851, 'lambda_l2': 0.0021202737014242865, 'max_depth': 8, 'min_child_samples': 15}
#Best RMSE for LGB:  90.92446329074565


xgb_params = {
    'learning_rate': 0.02730109828631519,
    'max_depth': 9,
    'subsample': 0.854847064173804,
    'colsample_bytree': 0.6421317014498971,
    'reg_alpha': 0.07672242008556412,
    'reg_lambda': 0.027970833737134406,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 2000,
    'random_state': 42,
    'n_jobs': -1
}

model_XGB = xgb.XGBRegressor(**xgb_params)

model_XGB.fit(
    X_train,
    y_train,
    verbose=100
        # eval_set=[(X_train, y_train), (X_val, y_val)],
)

y_train_pred = model_XGB.predict(X_train)
# # y_val_pred = model_XGB.predict(X_val)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
# # rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

print("XGBoost finished training.")
print(f"Train RMSE for XGB: {rmse_train:.4f}")
# # print(f"Validation RMSE for XGB: {rmse_val:.4f}")

# xgb_params = study.best_params
# xgb_params.update({
#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',
#     'n_estimators': 2000,
#     'random_state': 42,
#     'n_jobs': -1,
# })
# model_XGB = xgb.XGBRegressor(**xgb_params)

# model_XGB.fit(
#     X_train,
#     y_train,
#     verbose=100
# )

# y_train_pred = model_XGB.predict(X_train)
# # y_val_pred = model_XGB.predict(X_val)

# rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
# # rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
# print("XGBoost finished training.")
# print(f"Train RMSE for XGB: {rmse_train:.4f}")
# print(f"Validation RMSE for XGB: {rmse_val:.4f}")


lgb_params = {
    'learning_rate': 0.045699377930078715,
    'num_leaves': 88,
    'feature_fraction': 0.6898429945861433,
    'bagging_fraction': 0.8183989789261347,
    'bagging_freq': 5,
    'lambda_l1': 0.1928808129416336,
    'lambda_l2': 0.30364177014887417,
    'max_depth': 9,
    'min_child_samples': 11,
    'n_estimators': 2000,
    'random_state': 42,
    'verbosity': -1
}

model_lgb = lgb.LGBMRegressor(**lgb_params)

model_lgb.fit(
    X_train, y_train,
    callbacks=[
        lgb.log_evaluation(0) 
    ]         
           # eval_set=[(X_val, y_val)],
    # lgb.early_stopping(stopping_rounds=20),
)

y_train_pred = model_lgb.predict(X_train)
# y_val_pred = model_lgb.predict(X_val)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
# rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

print("LightGBM finished training.")
print(f"Train RMSE for LightGBM: {rmse_train:.4f}")
# print(f"Validation RMSE for LightGBM: {rmse_val:.4f}")

# lgb_params = study_lgb.best_params
# lgb_params.update({
#     'n_estimators': 2000,
#     'random_state': 42,
#     'verbosity': -1,
# })

# model_lgb = lgb.LGBMRegressor(**lgb_params)
# model_lgb.fit(
#     X_train, y_train,
#     callbacks=[lgb.log_evaluation(0)]
# )

# y_train_pred = model_lgb.predict(X_train)
# # y_val_pred = model_lgb.predict(X_val)

# rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
# # rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
# print("LightGBM finished training.")
# print(f"Train RMSE for LightGBM: {rmse_train:.4f}")
# # print(f"Validation RMSE for LightGBM: {rmse_val:.4f}")


# test_ids = df_test['id']
df_test = df_test.drop('id', axis=1)
prediction_xgb = model_XGB.predict(df_test)
# prediction_xgb = model_lgb.predict(df_test_no_text)
submission = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")

assert len(prediction_xgb) == submission.shape[0], "rows does no match"

submission['price'] = prediction_xgb
submission.to_csv("xgb.csv", index=False)
# submission.to_csv("xgb_notext.csv", index=False)


prediction_lgb = model_lgb.predict(df_test)
# prediction_lgb = model_lgb.predict(df_test_no_text)
submission = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")

assert len(prediction_lgb) == submission.shape[0], "rows does no match"

submission['price'] = prediction_lgb
submission.to_csv("lgb.csv", index=False)
# submission.to_csv("lgb_notext.csv", index=False)


submission = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv")
submission['price'] = prediction_lgb*0.5 + prediction_xgb*0.5
submission.to_csv("combine.csv", index=False)
# submission.to_csv("combine_notext.csv", index=False)


num_cols2 = df_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()

plt.figure(figsize=(16, 10))
sns.heatmap(df_train[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Antar Variabel Kuantitatif')
plt.show()


df_train.info()

