import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor 
from sklearn.model_selection import train_test_split
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import statsmodels.api as sm
import holidays
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')


train.head()


train.info()


train.isnull().sum()


train['date'] = pd.to_datetime(train['date'])  
test['date'] = pd.to_datetime(test['date'])
train['year'] = train['date'].dt.year.astype('int')  
test['year'] = test['date'].dt.year.astype('int')  


train_df_imputed = train.copy()


years = [str(year) for year in range(2010, 2021)]
gdp_per_capita_filtered_df = gdp_per_capita.loc[
    gdp_per_capita["Country Name"].isin(train["country"].unique()),
    ["Country Name"] + years
].set_index("Country Name")


for year in years:
    gdp_per_capita_filtered_df[f"{year}_ratio"] = gdp_per_capita_filtered_df[year] / gdp_per_capita_filtered_df[year].sum()

gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_df[[f"{year}_ratio" for year in years]]
gdp_per_capita_filtered_ratios_df.columns = [int(year) for year in years]
gdp_per_capita_filtered_ratios_df = (
    gdp_per_capita_filtered_ratios_df
    .unstack()
    .reset_index()
    .rename(columns={"level_0": "year", 0: "ratio", "Country Name": "country"})
)
gdp_per_capita_filtered_ratios_df["year"] = gdp_per_capita_filtered_ratios_df["year"].astype(int)


def impute_values(country, store, product, year, target_country, ratio):
    target_values = train_df_imputed.loc[
        (train_df_imputed["country"] == target_country) &
        (train_df_imputed["store"] == store) &
        (train_df_imputed["product"] == product) &
        (train_df_imputed["year"] == year),
        "num_sold"
    ]

    if not target_values.empty:
        target_value = target_values.values[0]  
        train_df_imputed.loc[
            (train_df_imputed["country"] == country) &
            (train_df_imputed["store"] == store) &
            (train_df_imputed["product"] == product) &
            (train_df_imputed["year"] == year) &
            (train_df_imputed["num_sold"].isna()),
            "num_sold"
        ] = target_value * ratio

for year in train_df_imputed["year"].unique():
    norway_ratio = gdp_per_capita_filtered_ratios_df.loc[
        (gdp_per_capita_filtered_ratios_df["year"] == year) & 
        (gdp_per_capita_filtered_ratios_df["country"] == "Norway"), 
        "ratio"
    ].values[0]

    for country in ["Canada", "Kenya"]:
        country_ratio = gdp_per_capita_filtered_ratios_df.loc[
            (gdp_per_capita_filtered_ratios_df["year"] == year) & 
            (gdp_per_capita_filtered_ratios_df["country"] == country), 
            "ratio"
        ].values[0]

        ratio = country_ratio / norway_ratio

        store_product_combinations = [
            ("Discount Stickers", "Holographic Goose"),
            ("Premium Sticker Mart", "Holographic Goose"),
            ("Stickers for Less", "Holographic Goose"),
        ]
        if country == "Kenya":
            store_product_combinations.append(("Discount Stickers", "Kerneler"))

        for store, product in store_product_combinations:
            impute_values(country, store, product, year, "Norway", ratio)

train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195


train_df_imputed.head()


def transform_date(df, col):
    df['quarter'] = df[col].dt.quarter.astype('int')
    df['month'] = df[col].dt.month.astype('int')
    df['day'] = df[col].dt.day.astype('int')
    df['day_of_week'] = df[col].dt.dayofweek.astype('int')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('int')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df


train = transform_date(train_df_imputed, 'date')
test = transform_date(test, 'date')


train_df_imputed = train_df_imputed.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)


cat_cols = ['country','store','product']


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    train_df_imputed[col] = le.fit_transform(train_df_imputed[col])
    label_encoders[col] = le


train_df_imputed.head()


train_df_imputed = train_df_imputed.dropna()


train_df_imputed['num_sold'] = np.log1p(train_df_imputed['num_sold'])


X = train_df_imputed.drop(columns=['num_sold'])
y = train_df_imputed['num_sold']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'device': 'gpu',
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, y_pred)

#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)

#best_params = study.best_params
#print(f"Best parameters: {best_params}")


params = {
    'objective': 'regression',
    'metric': 'mape',
    'device': 'gpu',
    'n_jobs': -1,
    'n_estimators': 2836,
    'learning_rate': 0.09416914107144629,
    'max_depth': 12,
    'reg_alpha': 0.08115084211052784,
    'reg_lambda': 0.8814474754002811,
    'min_child_samples': 10,
    'colsample_bytree': 0.918737313780491,
    'subsample': 0.9619203547509582
}

model = lgb.LGBMRegressor(**params)
model.fit(X, y)


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


test.head()


submission_ids = test['id']
predictions = model.predict(test)


predictions = np.expm1(predictions)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

