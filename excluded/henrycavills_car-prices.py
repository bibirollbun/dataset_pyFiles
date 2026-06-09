import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
import optuna
import holidays
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append('/kaggle/input/russian-car-plates-prices-prediction')
from supplemental_english import REGION_CODES, GOVERNMENT_CODES


def calc_smape(actual, forecast):
    act = np.array(actual)
    frc = np.array(forecast)
    return 100 / len(act) * np.sum(2 * np.abs(frc - act) / (np.abs(act) + np.abs(frc)))


df_train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv').drop(columns=['id'])
df_test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv').drop(columns=['id'])


def parse_plate_data(df):
    df['num_part_A'] = df['plate'].str[0]
    df['num_part_B'] = df['plate'].str[4:6]
    df['full_series'] = df['num_part_A'] + df['num_part_B']
    df['reg_code'] = df['plate'].str[1:4].astype(int)
    df['region_code'] = df['plate'].str[6:]
    return df

df_train = parse_plate_data(df_train)
df_test = parse_plate_data(df_test)


region_mapping = {}
for region_name, code_list in REGION_CODES.items():
    for code_item in code_list:
        region_mapping[code_item] = region_name

df_train['region'] = df_train['region_code'].apply(lambda x: region_mapping[x])
df_test['region'] = df_test['region_code'].apply(lambda x: region_mapping[x])


govt_codes_dict = defaultdict(lambda: defaultdict(dict))
for key, value in GOVERNMENT_CODES.items():
    series, reg_range, code = key
    descr, forbidden, advantage, significance = value
    govt_codes_dict[series][range(reg_range[0], reg_range[1] + 1)][code] = [descr, forbidden, advantage, significance]


def assign_vehicle_priorities(df):
    priority_list = []
    for _, row in df[['full_series', 'reg_code', 'region_code']].iterrows():
        descr, forb, adv, signf, flag = "No Description", 0, 0, 0, 0
        series = row['full_series']
        reg = row['reg_code']
        region_num = row['region_code']
        if series in govt_codes_dict:
            for reg_range, mapping in govt_codes_dict[series].items():
                if reg in reg_range:
                    if region_num in mapping:
                        values = mapping[region_num]
                        descr, forb, adv, signf = values
                        flag = 1
                        break
        priority_list.append([forb, adv, signf, flag, descr])
    return pd.DataFrame(priority_list, columns=['forbidden_to_buy', 'has_advantage', 'significance_level', 'govt_vehicle', 'description'])

df_train[['forbidden_to_buy', 'has_advantage', 'significance_level', 'govt_vehicle', 'description']] = assign_vehicle_priorities(df_train)
df_test[['forbidden_to_buy', 'has_advantage', 'significance_level', 'govt_vehicle', 'description']] = assign_vehicle_priorities(df_test)


df_train['region_code'] = df_train['region_code'].astype(int)
df_test['region_code'] = df_test['region_code'].astype(int)


df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

def derive_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    min_dates = df.groupby('plate')['date'].transform('min')
    df['days_since_listed'] = (df['date'] - min_dates).dt.days
    df['months_since_listed'] = (df['days_since_listed'] / 30).round(3)
    df['years_since_listed'] = (df['months_since_listed'] / 12).round(3)
    df['is_year_end'] = (df['month'] == 12)
    df['listing_order'] = df.groupby('plate')['date'].rank(method='dense').astype(int)
    df['date'] = df['date'].dt.date
    return df

df_train = derive_time_features(df_train)
df_test = derive_time_features(df_test)
df_train.sort_values(by=['plate', 'date'], inplace=True)


model_train = df_train.drop(['date', 'plate'], axis=1)
model_test = df_test.drop(['date', 'plate'], axis=1)

model_train.info()
print(model_train.nunique())


cat_features = ['num_part_A', 'num_part_B', 'region', 'full_series', 'description']
combined = pd.concat([model_train[cat_features], model_test[cat_features]], axis=0)
ord_enc = OrdinalEncoder()
ord_enc.fit(combined)
model_train[cat_features] = ord_enc.transform(model_train[cat_features])
model_test[cat_features] = ord_enc.transform(model_test[cat_features])


plt.figure(figsize=(30, 20))
n_cols = 5
n_rows = -(-model_train.shape[1] // n_cols)
for idx, col in enumerate(model_train.columns):
    plt.subplot(n_rows, n_cols, idx + 1)
    sns.kdeplot(data=model_train, x=col)
plt.tight_layout()
plt.show()


def evaluate_model_cv(model, X_data, y_data, n_splits=5, seed=42):
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score_list = []
    for train_idx, val_idx in k_fold.split(X_data):
        X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_tr, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
        model.fit(X_tr, y_tr)
        predictions = model.predict(X_val)
        fold_score = calc_smape(np.expm1(y_val), np.expm1(predictions))
        score_list.append(fold_score)
    return np.mean(score_list)


def test_model_on_folds():
    X_features = model_train.drop(['price'], axis=1)
    y_target = np.log1p(model_train['price'])
    def evaluate_and_print(estimator, name):
        score = evaluate_model_cv(estimator, X_features, y_target)
        print(f'{name} regressor score: {score:.4f}')
    evaluate_and_print(CatBoostRegressor(verbose=False), 'CatBoost')
    evaluate_and_print(LGBMRegressor(verbose=-1), 'LightGBM')
    evaluate_and_print(XGBRegressor(), 'XGBoost')

test_model_on_folds()


features = model_train.drop(['price'], axis=1)
target = np.log1p(model_train['price'])
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)


final_model = CatBoostRegressor(verbose=False)
final_model.fit(X_train, y_train)
feat_imp = pd.DataFrame(list(zip(final_model.feature_importances_, final_model.feature_names_)), columns=['importance', 'feature'])
print(feat_imp.sort_values(by='importance', ascending=False).head())
params = {
    'iterations': 2538,
    'depth': 8,
    'learning_rate': 0.08470141619243111,
    'l2_leaf_reg': 0.006372137071839507,
    'random_strength': 0.29981761433181997,
    'bagging_temperature': 0.11869968385680696,
    'border_count': 223,
    'min_data_in_leaf': 50
}
final_model = CatBoostRegressor(**params, verbose=False)
cv_score = evaluate_model_cv(final_model, features, target)
print(f'CV SMAPE (CatBoost): {cv_score:.4f}')
final_model.fit(features, target)


test_features = model_test.drop(['price'], axis=1, errors='ignore')
test_preds = np.expm1(final_model.predict(test_features))
submission_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')
submission_df['price'] = test_preds
submission_df.to_csv('submission.csv', index=False)


top20 = df_train.sort_values(by='price', ascending=False).head(20)
print(top20[['plate', 'price']])

