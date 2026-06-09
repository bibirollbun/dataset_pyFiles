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


#import libraries:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import re

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool


from sklearn.model_selection import RandomizedSearchCV, KFold
import optuna

from imblearn.over_sampling import SMOTE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score





import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  

pd.set_option('display.max_columns', None)


train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv').drop(columns =['id'],axis =1 )
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv').drop(columns =['id'],axis =1 )


train.head()


test.head()


test.drop('price', axis =1, inplace=True)
test.head()


print(train.shape)
print(test.shape)


from supplemental_english import REGION_CODES as region_codes

region_codes


# Region code dictionary -> region code dataframe:

code_df = pd.DataFrame(list(region_codes.items()), columns=['Region', 'region_code'])

# Step 2: Explode the list into separate rows
code_df = code_df.explode('region_code', ignore_index=True)

# Step 2: Flatten the list in 'Code' column
code_df['region_code'] = code_df['region_code'].apply(lambda x: x[0] if isinstance(x, list) else x)
code_df.head()


#example:
code_df[code_df['Region'] == 'Moscow']


#Train Data 
train.isna().sum()


#Test Data 
test.isna().sum()


train['plate'].value_counts()


train[train['plate']== 'A949MP190']


def extract_plate_info(df):
    df['series1'] = df['plate'].apply(lambda x: x[0])
    df['series2'] = df['plate'].apply(lambda x: x[4:6])
    
    df['regist_code'] = df['plate'].apply(lambda x: x[1:4]).astype(int)
    df['region_code'] = df['plate'].apply(lambda x: x[6:])

    df['comb_series'] = df['series1'] + df['series2']

    return df

train = extract_plate_info(train)
test = extract_plate_info(test)

train.head()


#merge with region code:

train_data = pd.merge(train, code_df, on='region_code', how = 'left')
test_data = pd.merge(test, code_df, on='region_code', how = 'left')
train_data.head()


train_data['Region'].isnull().sum()


test_data['Region'].isnull().sum()


train_data.head()


from supplemental_english import  GOVERNMENT_CODES

GOVERNMENT_CODES_ = defaultdict(lambda: defaultdict(dict))
for key, value in GOVERNMENT_CODES.items():
    region, r, code = key
    desc, forbidden, advantage, sign = value
    GOVERNMENT_CODES_[region][range(r[0],r[1]+1)][code] = [desc, forbidden, advantage, sign]




# Step 1: Create nested dict GOVERNMENT_CODES2
GOVERNMENT_CODES2 = defaultdict(lambda: defaultdict(dict))

for key, value in GOVERNMENT_CODES.items():
    region, r, code = key  # unpack the tuple from the key
    description, forbidden, advantage, significance = value
    number_range = range(r[0], r[1] + 1)
    GOVERNMENT_CODES2[region][number_range][code] = [description, forbidden, advantage, significance]

# Step 2: Function to add priority columns
def add_priorities(df):
    priorities = []

    def govt_vehicles(row):
        series = row['comb_series']
        register_code = row['regist_code']
        region_code = row['region_code']

        # Default values
        description, forbidden, advantage, significance, govt = 'None', 0, 0, 0, 0

        if series in GOVERNMENT_CODES2:
            code_ranges = GOVERNMENT_CODES2[series]
            for reg_range in code_ranges:
                if register_code in reg_range:
                    region_dict = code_ranges[reg_range]
                    if region_code in region_dict:
                        values = region_dict[region_code]
                        description, forbidden, advantage, significance = values
                        govt = 1
                        break  # match found

        priorities.append([forbidden, advantage, significance, govt, description])

    # Apply the inner function across the DataFrame
    df[['comb_series', 'regist_code', 'region_code']].apply(govt_vehicles, axis=1)

    # Convert results into a DataFrame
    res_df = pd.DataFrame(priorities, columns=[
        'forbidden_to_buy', 'advantage', 'significance', 'govt_vehicle', 'description'
    ])
    return res_df

# Step 3: Add these columns to your train/test data
train_data = train_data.copy()
test_data = test_data.copy()

train_add = add_priorities(train_data)
test_add = add_priorities(test_data)

train_data = pd.concat([train_data, train_add], axis=1)
test_data = pd.concat([test_data, test_add], axis=1)



train_data.head()


train_data.shape


train_data['date'] = pd.to_datetime(train_data['date'])
train_data['year'] = train_data['date'].dt.year
#train_data['year_month'] = train_data['date'].dt.strftime('%Y-%m')

train_data['month'] = train_data['date'].dt.month
train_data['quarter'] = train_data['date'].dt.quarter
train_data['dayofweek'] = train_data['date'].dt.dayofweek
train_data['is_weekend'] = train_data['dayofweek'].isin([5, 6]).astype(int)
train_data['yyyy_mm'] = train_data['date'].dt.to_period('M').astype(str)

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['year'] = test_data['date'].dt.year
#test_data['year_month'] = test_data['date'].dt.strftime('%Y-%m')

test_data['month'] = test_data['date'].dt.month
test_data['quarter'] = test_data['date'].dt.quarter
test_data['dayofweek'] = test_data['date'].dt.dayofweek
test_data['is_weekend'] = test_data['dayofweek'].isin([5, 6]).astype(int)
test_data['yyyy_mm'] = test_data['date'].dt.to_period('M').astype(str)

train_data.head()




train_data['plate_length'] = train_data['plate'].str.len()
test_data['plate_length'] = test_data['plate'].str.len()

train_data.head()


train_data['reg_avg_price']=train_data.groupby('Region')['price'].transform('mean')



train_data.head()


train_data['reg_code_length'] = train_data['region_code'].str.len()
test_data['reg_code_length'] = test_data['region_code'].str.len()
train_data.head()


sns.histplot(np.log(train_data['price']), kde = True)
plt.show()


# Target outliers:

def detect_outlier_percentages(df):
    numeric_cols = df.select_dtypes(include='number').columns
    outlier_percentages = {}

    
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_mask = (df['price'] < lower) | (df['price'] > upper)
    outlier_percentage = 100 * outlier_mask.sum() / len(df)
    outlier_percentages['price'] = round(outlier_percentage, 2)

    return pd.Series(outlier_percentages).sort_values(ascending=False)

detect_outlier_percentages(train_data)



# First_letter:

pp = train_data.groupby('series1')['price'].mean().reset_index()
pp= pp.sort_values(by = 'price')

plt.figure(figsize=(20,5))
sns.barplot(x=pp['series1'], y=pp['price'])
plt.xlabel('series1/ First letter in plates')
plt.ylabel('Average price')
plt.show()


pp2 = train_data.groupby('series2')['price'].mean().reset_index()
pp2= pp2.sort_values(by = 'price')

plt.figure(figsize=(20,5))
sns.barplot(x=pp2['series2'], y=pp2['price'])
plt.xlabel('series2 letters')
plt.ylabel('Average price')
plt.xticks(rotation =90)
plt.show()


# First number:

oo = train_data.groupby('regist_code')['price'].mean().reset_index()
oo= oo.sort_values(by = 'price')

plt.figure(figsize=(20,5))
sns.barplot(x=oo['regist_code'], y=oo['price'])
plt.xlabel('registered code in the plate')
plt.ylabel('Average price')
plt.show()


ooo = train_data.groupby('region_code')['price'].mean().reset_index()
ooo= ooo.sort_values(by = 'price')

plt.figure(figsize=(20,5))
sns.barplot(x=ooo['region_code'], y=ooo['price'])
plt.xlabel('region code in the plate')
plt.ylabel('Average price')
plt.show()


#code length

no = train_data.groupby('reg_code_length')['price'].mean().reset_index()
no= no.sort_values(by = 'price')

sns.barplot(x=no['reg_code_length'], y=no['price'])
plt.xlabel('Code length')
plt.ylabel('Average price')
plt.show()


year_avg = train_data.groupby('year')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=year_avg['year'], y=year_avg['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('Year')
plt.show()


price_avg = train_data.groupby('yyyy_mm')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.scatterplot(x=price_avg['yyyy_mm'], y=price_avg['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('Year-month')
plt.show()


month_avg = train_data.groupby('month')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=month_avg['month'], y=price_avg['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('month')
plt.show()


q_avg = train_data.groupby('quarter')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=q_avg['quarter'], y=price_avg['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('Quarter')
plt.show()


q_avg = train_data.groupby('dayofweek')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=q_avg['dayofweek'], y=price_avg['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('Day of week')
plt.show()


dow = train_data.groupby('is_weekend')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=dow['is_weekend'], y=dow['price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price')
plt.xlabel('weekend?')
plt.show()


train_data = train_data.sort_values(by = 'reg_avg_price')

plt.figure(figsize=(15,5))
sns.scatterplot(x=train_data['Region'], y=train_data['reg_avg_price'])
plt.xticks(rotation = 90)
plt.ylabel('Average Price for each region')
plt.xlabel('Region')
plt.show()


xx=train_data['plate_length'].value_counts()

sns.barplot(x=xx.index, y=xx.values)
plt.xlabel('Plate Length')
plt.ylabel('Count')
plt.show()


zz= train_data.groupby('plate_length')['price'].mean().reset_index()

sns.barplot(x=zz['plate_length'], y=zz['price'])
plt.xlabel('Plate Length')
plt.ylabel('Average price')
plt.show()


gov = train_data.groupby('govt_vehicle')['price'].mean().reset_index()

sns.barplot(x=gov['govt_vehicle'], y=gov['price'])
plt.xlabel('Govt vehicle?')
plt.ylabel('Average price')
plt.show()


adv = train_data.groupby('advantage')['price'].mean().reset_index()

sns.barplot(x=adv['advantage'], y=adv['price'])
plt.xlabel('Advantage?')
plt.ylabel('Average price')
plt.show()


sig = train_data.groupby('significance')['price'].mean().reset_index()

plt.figure(figsize=(15,5))
sns.barplot(x=sig['significance'], y=sig['price'])
plt.xlabel('Significance Level')
plt.ylabel('Average price')
plt.show()


train_data.head()


# Option A: Convert to an integer representing number of months since a fixed point
train_data['date_index'] = (train_data['date'].dt.year - train_data['date'].dt.year.min()) * 12 + train_data['date'].dt.month
test_data['date_index'] = (test_data['date'].dt.year - test_data['date'].dt.year.min()) * 12 + test_data['date'].dt.month


train_data.head()


#Area code and city:


# Combine area_code and city_name into a new column
train_data['code_city'] = train_data['region_code'].astype(str) + "_" + train_data['Region'].astype(str)
test_data['code_city'] = test_data['region_code'].astype(str) + "_" + test_data['Region'].astype(str)

train_data.head()



train_data.head()


train_data.drop(['plate',   'reg_avg_price', 'year', 'dayofweek', 'is_weekend', 'month', 'quarter', 'advantage', 'forbidden_to_buy'], axis =1, inplace=True)
test_data.drop(['plate', 'year',   'dayofweek', 'is_weekend', 'month', 'quarter', 'advantage', 'forbidden_to_buy'], axis=1, inplace=True)


train_data.head()


train_data.dtypes


# Data Type convert:
train_data['regist_code'] = train_data['regist_code'].astype('category')
train_data['series1'] = train_data['series1'].astype('category')
train_data['series2'] = train_data['series2'].astype('category')
train_data['region_code'] = train_data['region_code'].astype('category')
train_data['comb_series'] = train_data['comb_series'].astype('category')
train_data['Region'] = train_data['Region'].astype('category')
train_data['significance'] = train_data['significance'].astype('category')
train_data['govt_vehicle'] = train_data['govt_vehicle'].astype('category')
train_data['description'] = train_data['description'].astype('category')
train_data['code_city'] = train_data['code_city'].astype('category')

test_data['regist_code'] = test_data['regist_code'].astype('category')
test_data['series1'] = test_data['series1'].astype('category')
test_data['series2'] = test_data['series2'].astype('category')
test_data['region_code'] = test_data['region_code'].astype('category')
test_data['comb_series'] = test_data['comb_series'].astype('category')
test_data['Region'] = test_data['Region'].astype('category')
test_data['significance'] = test_data['significance'].astype('category')
test_data['govt_vehicle'] = test_data['govt_vehicle'].astype('category')
test_data['description'] = test_data['description'].astype('category')
test_data['code_city'] = test_data['code_city'].astype('category')




train_encoded = train_data.copy()
test_encoded = test_data.copy()

# col = ['Code','first_letter']

# # 1. Compute frequency encoding only from train data
# for col in col:
#     freq = train_data[col].value_counts(normalize=True)

#     # 2. Map to train and test using the same mapping
#     train_encoded[col] = train_data[col].map(freq)
#     test_encoded[col] = test_data[col].map(freq).fillna(0)  # fill unseen codes in test with 0


# region_freq = train_data['Region'].value_counts(normalize=True).to_dict()
# train_encoded['region_freq'] = train_data['Region'].map(region_freq)
# test_encoded['region_freq'] = test_data['Region'].map(region_freq)



# train_encoded['code_city'] = train_encoded['code_city'].astype('category')
# test_encoded['code_city'] = test_encoded['code_city'].astype('category')

# train_encoded['year_month'] = train_encoded['year_month'].astype('category')
# test_encoded['year_month'] = test_encoded['year_month'].astype('category')


# train_encoded['first_letter'] = train_encoded['first_letter'].astype('category')
# test_encoded['first_letter'] = test_encoded['first_letter'].astype('category')



train_encoded.head()


train_encoded.dtypes


X = train_encoded.drop(columns= 'price')


y = train_encoded['price']

# Apply natural log transform
#y_log = np.log1p(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # handle divide-by-zero
    return np.mean(diff) * 100



# Define categorical columns
categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns.tolist()


kf = KFold(n_splits=5, shuffle=True, random_state=42)
smape_scores = []
oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"--- Fold {fold} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Apply log1p transformation to the target ---
    y_train_log = np.log1p(y_train)
   

    # --- Create Pool objects with categorical features ---
    train_pool = Pool(X_train, y_train_log, cat_features=categorical_columns)
    val_pool = Pool(X_val, y_val, cat_features=categorical_columns)  # No target here

    # --- Define CatBoost model ---
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg= 3.4,
        bagging_temperature= 0.34,
        random_strength= 1.43,
        border_count= 50,
        loss_function='MAE',
        task_type='CPU',
        verbose=0,
        random_state=42
    )



    # --- Train the model ---
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True)

    # --- Predict and reverse transform ---
    # y_pred_log = model.predict(X_val)
    # y_pred = np.expm1(y_pred_log)  # Back to original scale
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    score = smape(y_val.values, y_pred)


    oof_preds[val_idx] = y_pred

    # --- Evaluate using SMAPE ---
    score = smape(y_val.values, y_pred)
    smape_scores.append(score)
    print(f"Fold {fold} SMAPE: {score:.2f}%")

# --- Final result ---
print(f"\nAverage SMAPE across folds: {np.mean(smape_scores):.2f}% ± {np.std(smape_scores):.2f}%")



# # Initialize the SHAP explainer

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test) 

#  Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot
shap.summary_plot(shap_values, X_test)


# === Inference on test set ===
test_pool = Pool(test_encoded, cat_features=categorical_columns)
test_pred_log = model.predict(test_pool)
test_pred = np.expm1(test_pred_log)  # Reverse the log1p
test_pred = np.clip(test_pred, 0, None)






# Clip negative values to 0
#test_pred = np.maximum(0, test_pred)

submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')
submission['price'] = test_pred
submission




#submitting results
submission.to_csv('submission.csv', index=False)



#Test Prediction values 


plt.figure(figsize=(6,4))
plt.hist(test_pred, bins=100)
plt.title("Test Predictions")
plt.show()


ls

