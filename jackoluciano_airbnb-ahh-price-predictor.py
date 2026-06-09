import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ast

# load data
train = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/train.csv")
test = pd.read_csv("/kaggle/input/sparta-2024-data-science-competition/test.csv")


train['first_review'] = pd.to_datetime(train['first_review'], errors='coerce')
train['last_review'] = pd.to_datetime(train['last_review'], errors='coerce')
test['first_review'] = pd.to_datetime(test['first_review'], errors='coerce')
test['last_review'] = pd.to_datetime(test['last_review'], errors='coerce')

# hitung selisih hari
train['review_gap_days'] = (train['last_review'] - train['first_review']).dt.days
test['review_gap_days'] = (test['last_review'] - test['first_review']).dt.days


drop_cols = ['id', 'name', 'description', 'neighborhood_overview', 'host_id',
             'host_url', 'host_name', 'host_about', 'host_location',
             'host_neighbourhood', 'host_verifications', 'first_review',
             'last_review', 'host_since']
train.drop(columns=[col for col in drop_cols if col in train.columns], inplace=True)
test.drop(columns=[col for col in drop_cols if col in test.columns], inplace=True)


bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability']
for col in bool_cols:
    train[col] = train[col].map({'t': 1, 'f': 0})
    test[col] = test[col].map({'t': 1, 'f': 0})


train['host_response_time'] = train['host_response_time'].astype(str)
test['host_response_time'] = test['host_response_time'].astype(str)
resp_all = pd.concat([train['host_response_time'], test['host_response_time']])
resp_map = {v: i for i, v in enumerate(resp_all.unique())}
train['host_response_time'] = train['host_response_time'].map(resp_map)
test['host_response_time'] = test['host_response_time'].map(resp_map)


def convert_percent(x):
    try:
        return float(x.strip('%')) / 100
    except:
        return np.nan

train['host_response_rate'] = train['host_response_rate'].map(convert_percent)
test['host_response_rate'] = test['host_response_rate'].map(convert_percent)
train['host_acceptance_rate'] = train['host_acceptance_rate'].map(convert_percent)
test['host_acceptance_rate'] = test['host_acceptance_rate'].map(convert_percent)


def extract_bathroom_number(x):
    try:
        return float(str(x).split(' ')[0])
    except:
        return np.nan

train['bathrooms'] = train['bathrooms_text'].map(extract_bathroom_number)
test['bathrooms'] = test['bathrooms_text'].map(extract_bathroom_number)


# parsing jadi list
def extract_amenities_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

train['amenities_list'] = train['amenities'].map(extract_amenities_list)
test['amenities_list'] = test['amenities'].map(extract_amenities_list)

# ambil top 50 amenities
from collections import Counter

all_amenities = train['amenities_list'].explode().tolist() + test['amenities_list'].explode().tolist()
amenity_counts = Counter(all_amenities)
top_amenities = [a for a, _ in amenity_counts.most_common(50)]

# vektor boolean (multi hot encode)
for amenity in top_amenities:
    col = f'amenity_{amenity}'
    train[col] = train['amenities_list'].apply(lambda x: int(amenity in x))
    test[col] = test['amenities_list'].apply(lambda x: int(amenity in x))

# drop kolom yg sudah diganti
train.drop(columns=['amenities', 'amenities_list'], inplace=True)
test.drop(columns=['amenities', 'amenities_list'], inplace=True)


train['neighbourhood'] = train['neighbourhood'].astype(str)
test['neighbourhood'] = test['neighbourhood'].astype(str)
neigh_all = pd.concat([train['neighbourhood'], test['neighbourhood']])
neigh_map = {v: i for i, v in enumerate(neigh_all.unique())}
train['neighbourhood'] = train['neighbourhood'].map(neigh_map)
test['neighbourhood'] = test['neighbourhood'].map(neigh_map)


# Encode fitur kategorikal lainnya
cat_cols = ['property_type', 'room_type', 'neighbourhood_cleansed', 'city']
for col in cat_cols:
    full = pd.concat([train[col], test[col]], axis=0)
    mapping = {k: v for v, k in enumerate(full.astype(str).unique())}
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)


train.drop(columns=['bathrooms_text', 'amenities'], inplace=True, errors='ignore')
test.drop(columns=['bathrooms_text', 'amenities'], inplace=True, errors='ignore')


train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)


# pisahkan target
target = train['price']
train.drop(columns=['price'], inplace=True)

# untuk memastikan
used_features = train.select_dtypes(include=[np.number]).columns.tolist()
X_train = train[used_features]
X_test = test[used_features]
y_train = np.log1p(target)


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)


# Model dan training
model = xgb.XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.08)
model.fit(X_tr, y_tr)

# Evaluasi
val_preds = model.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))

# Prediksi ke test
y_pred = model.predict(X_test)

# Load sample submission
submission = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/sample_submission.csv')

# Tulis prediksi ke kolom 'price'
submission['price'] = np.expm1(y_pred)

# Save ke CSV
submission.to_csv('/kaggle/working/sample_submission.csv', index=False)

