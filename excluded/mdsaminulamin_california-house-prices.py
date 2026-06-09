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


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train_df = pd.read_csv('/kaggle/input/california-house-prices/train.csv')
test_df = pd.read_csv('/kaggle/input/california-house-prices/test.csv')


train_df.shape


train_df.describe()


train_df.info()


train_df.isnull().sum().sort_values(ascending=False)


num_features = train_df.select_dtypes(include=['int64', 'float64']).columns
cat_features = train_df.select_dtypes(include=['object']).columns

print("Numeric:", len(num_features))
print("Categorical:", len(cat_features))


sns.histplot(train_df['Sold Price'], bins=50)
plt.show()


train_df[num_features].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


train_df[num_features].corr()['Sold Price'].sort_values(ascending=False)


for col in cat_features:
    print(col, train_df[col].nunique())


sns.boxplot(x='Bedrooms', y='Sold Price', data=train_df)


train_df.groupby('Type')['Sold Price'].median().sort_values()


train = train_df.copy()
test = test_df.copy()


train.head()


train['Sold Price'].describe()


plt.figure(figsize=(8,4))
sns.histplot(train['Sold Price'], bins=80, kde=True)
plt.show()


train['target_log1p'] = np.log1p(train['Sold Price'])


plt.figure(figsize=(8,4))
sns.histplot(train['target_log1p'], bins=80, kde=True)
plt.show()


numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()


numeric_cols


numeric_cols.remove('Sold Price')
numeric_cols.remove('target_log1p')


numeric_cols


cat_cols


corrs = train[numeric_cols + ['target_log1p']].corr()['target_log1p'].sort_values(ascending=False)


for col in ['Listed On','Last Sold On']:
    if col in train.columns:
        train[col] = pd.to_datetime(train[col], errors='coerce')
    if col in test.columns:
        test[col] = pd.to_datetime(test[col], errors='coerce')


# GPT given
for df in [train, test]:
    if 'Listed On' in df.columns:
        df['listed_year'] = df['Listed On'].dt.year
        df['listed_month'] = df['Listed On'].dt.month
        df['listed_dayofweek'] = df['Listed On'].dt.dayofweek
        max_date = df['Listed On'].max()
        df['days_since_listed'] = (max_date - df['Listed On']).dt.days
    
    
    if 'Last Sold On' in df.columns:
        df['last_sold_year'] = df['Last Sold On'].dt.year
        df['last_sold_month'] = df['Last Sold On'].dt.month
        df['days_since_last_sold'] = (pd.to_datetime('today') - df['Last Sold On']).dt.days


for df in [train, test]:
    if 'Summary' in df.columns:
        df['summary_len'] = df['Summary'].fillna('').str.len()
        df['summary_word_count'] = df['Summary'].fillna('').str.split().apply(len)


# GPT Given
# Bedrooms cleaning: many datasets have 'Bedrooms' as object (e.g., 'Studio', '3')
def parse_bedrooms(x):
    if pd.isnull(x):
        return np.nan
    try:
        # remove plus signs, convert to float
        if isinstance(x, (int, float)):
            return x
        s = str(x).lower().strip()
        if s in ['studio']:
            return 0.5
        # sometimes contains '3 bd' etc. extract number
        import re
        m = re.search(r"\d+(?:\.\d+)?", s)
        if m:
            return float(m.group())
        return np.nan
    except Exception:
        return np.nan


for df in [train, test]:
    if 'Bedrooms' in df.columns:
        df['bedrooms_num'] = df['Bedrooms'].apply(parse_bedrooms)


# Bathrooms cleaning
for df in [train, test]:
    if 'Bathrooms' in df.columns:
        df['bathrooms_num'] = pd.to_numeric(df['Bathrooms'], errors='coerce')
    if 'Full bathrooms' in df.columns:
        df['full_bathrooms'] = pd.to_numeric(df['Full bathrooms'], errors='coerce')


# Lot: could be area described like '10,000 sqft' -> attempt numeric extraction
import re

def extract_numeric(x):
    if pd.isnull(x):
        return np.nan
    try:
        s = str(x)
        # remove commas
        s_clean = re.sub(r'[,$]','', s)
        m = re.search(r"\d+\.?\d*", s_clean)
        if m:
            return float(m.group())
        return np.nan
    except Exception:
        return np.nan


for df in [train, test]:
    if 'Lot' in df.columns:
        df['lot_num'] = df['Lot'].apply(extract_numeric)


# Garage spaces, Total interior livable area - ensure numeric
for df in [train, test]:
    if 'Garage spaces' in df.columns:
        df['garage_spaces_num'] = pd.to_numeric(df['Garage spaces'], errors='coerce')
    if 'Total interior livable area' in df.columns:
        df['total_interior_area'] = pd.to_numeric(df['Total interior livable area'], errors='coerce')
    if 'Total spaces' in df.columns:
        df['total_spaces_num'] = pd.to_numeric(df['Total spaces'], errors='coerce')
    if 'Year built' in df.columns:
        df['year_built_num'] = pd.to_numeric(df['Year built'], errors='coerce')


# Create a few interaction features
for df in [train, test]:
    df['area_per_bed'] = df['total_interior_area'] / (df['bedrooms_num'].replace(0, np.nan))
    df['price_per_sqft'] = np.nan # placeholder - only compute on train later using Sold Price


candidate_numeric = [
    'listed_year',
    'listed_month',
    'listed_dayofweek',
    'days_since_listed',
    'summary_len',
    'summary_word_count',
    'bedrooms_num',
    'bathrooms_num',
    'full_bathrooms',
    'lot_num',
    'garage_spaces_num',
    'total_interior_area',
    'total_spaces_num',
    'year_built_num',
    'area_per_bed'
]

candidate_categorical = [
    'Type',
    'Region',
    'City',
    'Zip',
    'Heating',
    'Cooling'
]


candidate_numeric


candidate_categorical


train["price_per_sqft"] = train['Sold Price'] / train["total_interior_area"].replace(0, np.nan)
train["log_price_per_sqft"] = np.log1p(train["price_per_sqft"])


low_card_cat = []
high_card_cat = []
for c in candidate_categorical:
    nun = train[c].nunique()
    if nun <= 20:
        low_card_cat.append(c)
    else:
        high_card_cat.append(c)


low_card_cat


high_card_cat


train_proc = train.copy()
test_proc = test.copy()


class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, target_col='target_log1p', m=10):
        self.cols = cols
        self.target_col = target_col
        self.m = m
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if self.target_col in df.columns:
            target = df[self.target_col]
        else:
            if y is None:
                raise ValueError("Target missing")
            target = y
        
        self.global_mean_ = target.mean()

        for col in self.cols:
            stats = pd.DataFrame({
                'count': df.groupby(col)[self.target_col].count(),
                'mean': df.groupby(col)[self.target_col].mean()
            }).reset_index()

            stats["encoding"] = (
                (stats["count"] * stats["mean"] + self.m * self.global_mean_)
                / (stats["count"] + self.m)
            )
            self.encodings_[col] = stats.set_index(col)["encoding"].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        new_cols = []
        for col in self.cols:
            new_name = f"{col}_te"
            X[new_name] = X[col].map(self.encodings_[col]).fillna(self.global_mean_)
            new_cols.append(new_name)
        return X[new_cols]

encoder = SmoothTargetEncoder(cols=high_card_cat, target_col="target_log1p", m=20)
train_te = encoder.fit_transform(train_proc)
test_te = encoder.transform(test_proc)


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, candidate_numeric)
    ],
    remainder="drop"
)


X_num = preprocessor.fit_transform(train_proc)

X_num_df = pd.DataFrame(X_num, columns=candidate_numeric, index=train_proc.index)

X = pd.concat([X_num_df, train_te.reset_index(drop=True)], axis=1)

y = train_proc["target_log1p"]


X_test_num = preprocessor.transform(test_proc)
X_test_num_df = pd.DataFrame(X_test_num, columns=candidate_numeric, index=test_proc.index)

X_test = pd.concat([X_test_num_df, test_te.reset_index(drop=True)], axis=1)


X.shape


X_test.shape


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=300,
    task_type="CPU",
    l2_leaf_reg=1,
    bagging_temperature=1
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)



# param_grid = {
#     "depth": [6, 8, 10],
#     "learning_rate": [0.01, 0.03, 0.05],
#     "l2_leaf_reg": [1, 3, 5, 7],
#     "iterations": [1000, 2000],
#     "bagging_temperature": [0.5, 1, 2],
# }


# from catboost import CatBoostRegressor
# from sklearn.model_selection import GridSearchCV, KFold

# cat_model = CatBoostRegressor(
#     loss_function="RMSE",
#     eval_metric="RMSE",
#     random_seed=42,
#     verbose=0,
#     task_type="CPU"
# )


# kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# grid = GridSearchCV(
#     estimator=cat_model,
#     param_grid=param_grid,
#     cv=kfold,
#     scoring="neg_root_mean_squared_error",
#     n_jobs=-1,
#     verbose=3
# )

# grid.fit(X, y)


# print("Best Score:", -grid.best_score_)
# print("Best Params:", grid.best_params_)


from math import sqrt

rmse = sqrt(mean_squared_error(y_val, model.predict(X_val)))
rmse


# model.fit(X, y)


test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)


submission = pd.DataFrame({
    "Id": test["Id"],
    "Sold Price": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission saved!")




