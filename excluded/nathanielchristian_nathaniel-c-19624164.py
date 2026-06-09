import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv(r"/kaggle/input/sparta-2024-data-science-competition/train.csv")
# train_df = train_df.iloc[:, 1:]
test_df = pd.read_csv(r"/kaggle/input/sparta-2024-data-science-competition/test.csv")
# test_df = test_df.iloc[:, 1:]


display(train_df.head())

print(f"\nDimensions: {train_df.shape[0]} rows × {train_df.shape[1]} columns")


train_df['id'] = train_df['id'].astype('string')
train_df['host_id'] = train_df['host_id'].astype('string')

test_df['id'] = test_df['id'].astype('string')
test_df['host_id'] = test_df['host_id'].astype('string')


print("Null values in train_df:")
print(train_df.isnull().sum())


idtest = test_df['id']


def drop_cols(df):
    df.drop(columns=[
        'name',
        'description',
        'neighborhood_overview',
        'host_location',
        'host_about',
        'host_neighbourhood',
        'host_name',
        'availability_eoy',
        'number_of_reviews_ly',
        'estimated_occupancy_l365d',
        'estimated_revenue_l365d',
        'first_review',
        'last_review',
        'neighbourhood',
        'neighbourhood_cleansed',
        'review_scores_rating',
        'review_scores_accuracy',
        'review_scores_cleanliness',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_location',
        'review_scores_value',
        'reviews_per_month',
        'host_response_rate',
        'host_response_time',
        'host_acceptance_rate',
        'property_type',
        'id',
        'host_id',
        'amenities',
    ], inplace=True)
    return df


train_df = drop_cols(train_df)
test_df = drop_cols(test_df)


def fillna_with_median(df):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

train_df = fillna_with_median(train_df)
test_df = fillna_with_median(test_df)


def convert_host_since(df, ref_date="2025-07-31"):
    # Konversi ke datetime
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    ref_date = pd.to_datetime(ref_date)
    
    # Hitung jumlah bulan antara dua tanggal
    df['months_since'] = (
        (ref_date.year - df['host_since'].dt.year) * 12 +
        (ref_date.month - df['host_since'].dt.month)
    ).astype('Int64')
    
    # Hapus kolom asli
    df.drop(columns=['host_since'], inplace=True)
    return df

train_df = convert_host_since(train_df)
test_df = convert_host_since(test_df)


pd.set_option('display.max_columns', None)
display(train_df.head())


# # Numerical columns: calculate range
# numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
# for column in numerical_columns:
#     print(f"{column}: Range = {train_df[column].min()} to {train_df[column].max()}")

# # Categorical columns: list unique values
# categorical_columns = train_df.select_dtypes(include=['object', 'string']).columns
# for column in categorical_columns:
#     unique_values = train_df[column].unique()
#     print(f"{column}: Unique values = {', '.join(map(str, unique_values))}")


from ast import literal_eval
import re
from sklearn.preprocessing import StandardScaler

def encode_boolean_columns(df, cols):
    for col in cols:
        df[col] = df[col].map({'t': 1, 'f': 0})
    return df

def encode_host_verifications(df):
    df['host_verifications'] = df['host_verifications'].dropna().apply(literal_eval)
    
    all_verifs = set(v for sublist in df['host_verifications'].dropna() for v in sublist)
    
    for method in all_verifs:
        df[f'verified_{method}'] = df['host_verifications'].apply(
            lambda x: int(method in x) if isinstance(x, list) else 0
        )
    
    df.drop(columns=['host_verifications'], inplace=True)
    return df

def encode_room_type(df):
    dummies = pd.get_dummies(df['room_type'], prefix='room_type')
    df = pd.concat([df.drop(columns=['room_type']), dummies], axis=1)
    return df


def parse_bathrooms(df):
    def extract_bathroom_count(text):
        if pd.isna(text):
            return np.nan
        if 'Half-bath' in text:
            return 0.5
        match = re.match(r'(\d+(\.\d+)?)', text)
        return float(match.group(1)) if match else np.nan

    def is_shared(text):
        if pd.isna(text):
            return 0
        return int('shared' in text.lower())

    df['bathrooms'] = df['bathrooms_text'].apply(extract_bathroom_count)
    df['bathroom_is_shared'] = df['bathrooms_text'].apply(is_shared)
    df.drop(columns=['bathrooms_text'], inplace=True)
    return df


# check correlation between numerical columns

numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

correlation = train_df[numerical_cols].corr()


plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f",
            annot_kws={"size": 10}, xticklabels=True, yticklabels=True)

plt.tight_layout()
plt.title('Heatmap Korelasi Antar Variabel', fontsize=12,)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()


def fillna_numeric(df):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def fillna_categorical(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def fill_all_missing(df):
    df = fillna_numeric(df)
    df = fillna_categorical(df)
    return df


train_df = encode_boolean_columns(train_df, ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified','has_availability'])
train_df = encode_host_verifications(train_df)
train_df = encode_room_type(train_df)
train_df = parse_bathrooms(train_df)
train_df = fill_all_missing(train_df)

test_df = encode_boolean_columns(test_df, ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified','has_availability'])
test_df = encode_host_verifications(test_df)
test_df = encode_room_type(test_df)
test_df = parse_bathrooms(test_df)
test_df = fill_all_missing(test_df)


city_to_country = {
    'New York City': 'USA',
    'San Francisco': 'USA',
    'Chicago': 'USA',
    'Los Angeles': 'USA',
    'Boston': 'USA',
    'Austin': 'USA',
    'Seattle': 'USA',
    'Portland': 'USA',
    'Dallas': 'USA',
    'Denver': 'USA',
    'Oakland': 'USA',
    'Nashville': 'USA',
    'Newark': 'USA',
    'San Diego': 'USA',
    'Albany': 'USA',
    'Columbus': 'USA',
    'Fort Worth': 'USA',
    'Rochester': 'USA',
    'Salem, OR': 'USA',
    'Washington, D.C': 'USA',
    'San Mateo County': 'USA',
    'Santa Clara County': 'USA',
    'Santa Cruz County': 'USA',
    'Twin Cities MSA': 'USA',
    'Rhode Island': 'USA',
    'Broward County': 'USA',
    'Toronto': 'Canada',
    'Montreal': 'Canada',
    'Ottawa': 'Canada',
    'Winnipeg': 'Canada',
    'Quebec City': 'Canada',
    'New Brunswick': 'Canada',
    'London': 'UK',
    'Cambridge': 'UK',
    'Bristol': 'UK',
    'Greater Manchester': 'UK',
    'Edinburgh': 'UK',
    'Paris': 'France',
    'Lyon': 'France',
    'Bordeaux': 'France',
    'Berlin': 'Germany',
    'Munich': 'Germany',
    'Hamburg': 'Germany',
    'Madrid': 'Spain',
    'Barcelona': 'Spain',
    'Mallorca': 'Spain',
    'Menorca': 'Spain',
    'Valencia': 'Spain',
    'Lisbon': 'Portugal',
    'Porto': 'Portugal',
    'Rome': 'Italy',
    'Florence': 'Italy',
    'Naples': 'Italy',
    'Venice': 'Italy',
    'Milan': 'Italy',
    'Bologna': 'Italy',
    'Sicily': 'Italy',
    'Puglia': 'Italy',
    'Amsterdam': 'Netherlands',
    'Rotterdam': 'Netherlands',
    'The Hague': 'Netherlands',
    'Brussels': 'Belgium',
    'Antwerp': 'Belgium',
    'Ghent': 'Belgium',
    'Vienna': 'Austria',
    'Geneva': 'Switzerland',
    'Vaud': 'Switzerland',
    'Istanbul': 'Turkey',
    'Thessaloniki': 'Greece',
    'Athens': 'Greece',
    'Crete': 'Greece',
    'South Aegean': 'Greece',
    'Copenhagen': 'Denmark',
    'Oslo': 'Norway',
    'Dublin': 'Ireland',
    'Euskadi': 'Spain (Basque)',
    'Mexico City': 'Mexico',
    'Rio de Janeiro': 'Brazil',
    'Riga': 'Latvia',
    'Singapore': 'Singapore',
    'Bangkok': 'Thailand',
    'Taipei': 'Taiwan',
    'Hong Kong': 'Hong Kong',
    'Cape Town': 'South Africa',
    'Melbourne': 'Australia',
    'Sydney': 'Australia',
    'Brisbane': 'Australia',
    'Tasmania': 'Australia',
    'Western Australia': 'Australia',
    'Victoria': 'Australia',
    'Sunshine Coast': 'Australia',
    'Mornington Peninsula': 'Australia',
    'Mid North Coast': 'Australia',
    'Barwon South West, Vic': 'Australia',
    'Barossa Valley': 'Australia',
    'Belize': 'Belize',
    'Asheville': 'USA',
    'Bergamo': 'Italy',
    'Bozeman': 'USA',
    'Clark County, NV': 'USA',
    'Girona': 'Spain',
    'Hawaii': 'USA',
    'Ireland': 'Ireland',
    'Jersey City': 'USA',
    'Malta': 'Malta',
    'New Orleans': 'USA',
    'Northern Rivers': 'Australia',
    'Pacific Grove': 'USA',
    'Prague': 'Czech Republic',
    'Vancouver': 'Canada',
}

train_df['country'] = train_df['city'].map(city_to_country).fillna('Other')
train_df = train_df.drop(columns=['city'])

test_df['country'] = test_df['city'].map(city_to_country).fillna('Other')
test_df = test_df.drop(columns=['city'])



train_df.head()


major_countries = [
    'USA', 'Italy', 'Australia', 'Spain', 'UK', 'France', 'Greece', 'Canada',
    'Portugal', 'Brazil', 'Ireland', 'Germany', 'Malta', 'Mexico', 'Austria',
    'Belgium', 'Netherlands', 'Switzerland', 'Spain (Basque)', 'Denmark',
    'Hong Kong', 'Thailand', 'South Africa'
]

def map_country(country):
    if pd.isna(country):
        return 'Other'
    return country if country in major_countries else 'Other'


train_df['country_mapped'] = train_df['country'].apply(map_country)

train_df = pd.get_dummies(train_df, columns=['country_mapped'], prefix='country')
train_df = train_df.drop(columns=['country'])

test_df['country_mapped'] = test_df['country'].apply(map_country)

test_df = pd.get_dummies(test_df, columns=['country_mapped'], prefix='country')
test_df = test_df.drop(columns=['country'])


y = train_df['price']
train_df = train_df.drop(columns=['price'])


numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

binary_cols = [col for col in numerical_cols if set(train_df[col].unique()) <= {0, 1}]
numerical_cols_to_normalize = [col for col in numerical_cols if col not in binary_cols]

scaler = StandardScaler()
train_df[numerical_cols_to_normalize] = scaler.fit_transform(train_df[numerical_cols_to_normalize])

test_df[numerical_cols_to_normalize] = scaler.transform(test_df[numerical_cols_to_normalize])


train_df.head()


def smape(actual, forecast):
    value = 100 * np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast)))
    return round(float(value), 2)
def accuracy(y_true, y_pred):
    value = 100 - smape(y_true, y_pred)
    return round(float(value), 2)


X = train_df

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


xgb_train = xgb_model.predict(X_valid)
smape(y_valid, xgb_train)

mse = mean_squared_error(y_valid, xgb_train)
r2 = r2_score(y_valid, xgb_train)
rmse_ = rmse(y_valid, xgb_train)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("rmse Score:", rmse_)
print(accuracy(y_valid, xgb_train))


test_df


predictions = xgb_model.predict(test_df)


output_df = pd.DataFrame({
    'id': idtest,
    'price': predictions
})

output_df.to_csv('predicted_prices.csv', index=False)

print(output_df.head())

