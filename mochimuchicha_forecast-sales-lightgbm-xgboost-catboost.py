import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge


train_df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


print(f"Shape of training data: {train_df.shape}")
print(train_df.head())
print(f"\nShape of test data: {test_df.shape}")
print(test_df.head())


train_df.info()


test_ids = test_df['id']
target_column = 'num_sold'

categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

categorical_features = categorical_columns.tolist()
numerical_features = numerical_columns.tolist()

print("Categorical Columns: ", categorical_features)
print("Numerical Columns: ", numerical_features)

for column in categorical_columns:
    num_unique = train_df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


print(min(train_df['date']))
print(max(train_df['date']))
print(min(test_df['date']))
print(max(test_df['date']))


# Set up the figure
grid_size = (3, 1)
fig, axes = plt.subplots(3, 1, figsize=(12, 30))

# Number Sold by Country
sns.barplot(data=train_df, x="country", y="num_sold", errorbar=None, ax=axes[0])
axes[0].set_title("Number Sold by Country", fontsize=14)
axes[0].set_xlabel("Country", fontsize=12)
axes[0].set_ylabel("Number Sold", fontsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)

# Number Sold by Store
sns.barplot(data=train_df, x="store", y="num_sold", errorbar=None, ax=axes[1])
axes[1].set_title("Number Sold by Store", fontsize=14)
axes[1].set_xlabel("Store", fontsize=12)
axes[1].set_ylabel("Number Sold", fontsize=12)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

# Number Sold by Product
sns.barplot(data=train_df, x="product", y="num_sold", errorbar=None, ax=axes[2])
axes[2].set_title("Number Sold by Product", fontsize=14)
axes[2].set_xlabel("Product", fontsize=12)
axes[2].set_ylabel("Number Sold", fontsize=12)
axes[2].tick_params(axis='x', labelsize=12)
axes[2].tick_params(axis='y', labelsize=12)

# Adjust layout
plt.tight_layout()
plt.show()


print("Missing Values in Training Data")
print(train_df.isna().sum().sort_values(ascending=False))


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

train_df["year"] = train_df["date"].dt.year
train_df.head()


missing_df = train_df.loc[train_df['num_sold'].isna()]
missing_df_group = missing_df.groupby(['country', 'store', 'product']).size().reset_index(name='missing_count')
print(missing_df_group)


duplicates = train_df['id'].duplicated().sum()
print(f"Number of duplicate IDs: {duplicates}")


fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Histogram w/ KDE
sns.histplot(train_df['num_sold'], kde=True, bins=50, ax=axes[0])
axes[0].set_title("Histogram")
axes[0].set_xlabel('num_sold')
axes[0].set_ylabel('frequency')

# QQ Plot
sm.qqplot(train_df['num_sold'], line='45', ax=axes[1])
axes[1].set_title("QQ Plot")

plt.tight_layout()
plt.show()


gdp_per_capita_df = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')

years = [str(years) for years in range(2010, 2021)]

gdp_per_capita_df = gdp_per_capita_df.loc[
    gdp_per_capita_df["Country Name"].isin(train_df["country"].unique()),
    ["Country Name"] + years
].set_index("Country Name")

gdp_per_capita_df.head()


for year in years:
    gdp_per_capita_df[f"{year}_ratio"] = gdp_per_capita_df[year]/gdp_per_capita_df.sum()[year]

gdp_per_capita_ratio_df = gdp_per_capita_df[[f"{year}_ratio" for year in years]]
gdp_per_capita_ratio_df.columns = [int(year) for year in years]
gdp_per_capita_ratio_df = gdp_per_capita_ratio_df.unstack().reset_index().rename(
        columns={"level_0": "year", "Country Name": "country", 0:"ratio"}
)
gdp_per_capita_ratio_df["year"] = pd.to_datetime(gdp_per_capita_ratio_df["year"], format="%Y")
gdp_per_capita_ratio_df["year"] = gdp_per_capita_ratio_df["year"].dt.year

print(gdp_per_capita_ratio_df.head())


# loop through years to perform GDP-based imputations
for year in train_df["year"].unique():
    # retrieve GDP ratio for Norway and Canada
    target_ratio = gdp_per_capita_ratio_df.loc[
        (gdp_per_capita_ratio_df["year"]==year) &
        (gdp_per_capita_ratio_df["country"]=="Norway"), "ratio"
    ].values[0]
    
    current_ratio = gdp_per_capita_ratio_df.loc[
        (gdp_per_capita_ratio_df["year"]==year) &
        (gdp_per_capita_ratio_df["country"]=="Canada"), "ratio"
    ].values[0]

    ratio_can = current_ratio / target_ratio
        
    # impute missing num_sold for Canada
    # impute missing values in (Canada, Discount Stickers, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Canada") &
        (train_df["store"] == "Discount Stickers") & 
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]
    
    train_df.loc[
        (train_df["country"] == "Canada") &
        (train_df["store"] == "Discount Stickers") & 
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") &
            (train_df["store"] == "Discount Stickers") & 
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_can
    ).values

    # impute missing values in (Canada, Discount Stickers, Kerneler)
    current_ts = train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]
    
    train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Discount Stickers") &
            (train_df["product"] == "Kerneler") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_can
    ).values

    # impute missing values in (Canada, Premium Sticker Mart, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Premium Sticker Mart") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Premium Sticker Mart") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Premium Sticker Mart") &
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_can
    ).values

    # impute missing values in (Canada, Stickers for Less, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Stickers for Less") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Canada") & 
        (train_df["store"] == "Stickers for Less") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Stickers for Less") &
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_can
    ).values
    
    # retrieve GDP ratio for Norway and Kenya
    current_ratio = gdp_per_capita_ratio_df.loc[
        (gdp_per_capita_ratio_df["year"]==year) &
        (gdp_per_capita_ratio_df["country"]=="Kenya"), "ratio"
    ].values[0]

    ratio_ken = current_ratio / target_ratio
    ratio_ken -= 0.0007/2

    # impute missing num_sold for Kenya
    # impute missing values in (Kenya, Discount Stickers, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Discount Stickers") &
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_ken
    ).values

    # impute missing values in (Kenya, Discount Stickers, Kerneler)
    current_ts = train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Discount Stickers") &
            (train_df["product"] == "Kerneler") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_ken
    ).values

    # impute missing values in (Kenya, Discount Stickers, Kerneler Dark Mode)
    current_ts = train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler Dark Mode") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Discount Stickers") &
        (train_df["product"] == "Kerneler Dark Mode") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Discount Stickers") &
            (train_df["product"] == "Kerneler Dark Mode") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_ken
    ).values

    # impute missing values in (Kenya, Premium Sticker Mart, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Premium Sticker Mart") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Premium Sticker Mart") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Premium Sticker Mart") &
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_ken
    ).values
    
    # impute missing values in (Kenya, Stickers for Less, Holographic Goose)
    current_ts = train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Stickers for Less") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year)
    ]

    missing_ts_dates = current_ts.loc[
        current_ts["num_sold"].isna(), "date"
    ]

    train_df.loc[
        (train_df["country"] == "Kenya") & 
        (train_df["store"] == "Stickers for Less") &
        (train_df["product"] == "Holographic Goose") & 
        (train_df["year"] == year) &
        (train_df["date"].isin(missing_ts_dates)),
        "num_sold"
    ] = (
        train_df.loc[
            (train_df["country"] == "Norway") & 
            (train_df["store"] == "Stickers for Less") &
            (train_df["product"] == "Holographic Goose") & 
            (train_df["year"] == year) &
            (train_df["date"].isin(missing_ts_dates)),
            "num_sold"
        ] * ratio_ken
    ).values


print("Missing Values in Training Data")
print(train_df.isna().sum().sort_values(ascending=False))


train_df


store_ratio = train_df.groupby('store')['num_sold'].sum()/train_df['num_sold'].sum()

country_ratio = train_df.groupby('country')['num_sold'].sum()/train_df['num_sold'].sum()

product_df = train_df.groupby(["date", "product"])["num_sold"].sum().reset_index()
product_ratio = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio = product_ratio.apply(lambda x: x / x.sum(), axis=1)  # Normalize per day
product_ratio = product_ratio.stack().rename("ratios").reset_index()

print("Store Ratio:")
print(store_ratio)
print("\nCountry Ratio:")
print(country_ratio)
print("\nProduct Ratio (Daily):")
print(product_ratio)


def date(df):
    df['year'] = df['date'].dt.year.astype('float64')
    df['quarter'] = df['date'].dt.quarter.astype('float64')
    df['month'] = df['date'].dt.month.astype('float64')
    df['day'] = df['date'].dt.day.astype('float64')
    df['day_of_week'] = df['date'].dt.dayofweek.astype('float64')
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('float64')
    
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 365)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    return df

train_df = date(train_df)
test_df = date(test_df)


import holidays

def add_holiday_features(df):
    # Define holidays for relevant countries
    countries = df['country'].unique()
    holiday_data = {}
    for country in countries:
        if country == "Canada":
            holiday_data[country] = holidays.Canada()
        elif country == "Finland":
            holiday_data[country] = holidays.Finland()
        elif country == "Italy":
            holiday_data[country] = holidays.Italy()
        elif country == "Kenya":
            holiday_data[country] = holidays.Kenya()
        elif country == "Norway":
            holiday_data[country] = holidays.Norway()
        elif country == "Singapore":
            holiday_data[country] = holidays.Singapore()
    
    # Add holiday column
    df['holiday'] = df.apply(lambda row: int(row['date'] in holiday_data[row['country']]), axis=1)
    return df

# Add holiday features
train_df = add_holiday_features(train_df)
test_df = add_holiday_features(test_df)

# 1: holiday, 0: no holiday


train_df['num_sold'] = np.log1p(train_df['num_sold'])

fix, axes = plt.subplots(2, 1, figsize=(8, 10))

# Histogram w/ KDE
sns.histplot(train_df['num_sold'], kde=True, bins=50, ax=axes[0])
axes[0].set_title("Histogram")
axes[0].set_xlabel('num_sold')
axes[0].set_ylabel('frequency')

# QQ Plot
sm.qqplot(train_df['num_sold'], line='45', ax=axes[1])
axes[1].set_title("QQ Plot")

plt.tight_layout()
plt.show()


# Remove irrelevant columns
columns_to_drop = ['id', 'date']
target_column = 'num_sold'

X = train_df.drop(columns=columns_to_drop + [target_column])
y = train_df[target_column]
id_col = test_df['id']
test_df = test_df.drop(columns=columns_to_drop)


# Identify categorical and numerical features
train_categorical_columns = X.select_dtypes(include=['object']).columns
train_numerical_columns = X.select_dtypes(exclude=['object']).columns

test_categorical_columns = test_df.select_dtypes(include=['object']).columns

# Encoding categorical features
X = pd.get_dummies(X, columns=train_categorical_columns)
test_df = pd.get_dummies(test_df, columns=test_categorical_columns)


X.head()


y.head()


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
from lightgbm import LGBMRegressor

def objective_lgb(trial):
    # Parameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 10, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective': 'regression',
        'device': 'gpu',
        'metric': 'mape',
        'random_state': 42,
        'verbosity': -1
    }
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train LightGBM model
    model = LGBMRegressor(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # Predict and calculate MAPE
    y_val_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    return mape

# Optimize
#study_lgb = optuna.create_study(direction='minimize')
#study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

# Print best parameters
#best_params_lgb = study_lgb.best_params
#print("Best LightGBM Parameters:", best_params_lgb)
#print(f"Best MAPE: {study_lgb.best_value:.4f}")

best_params_lgb = {
    'n_estimators': 3770, 
    'learning_rate': 0.05038034487788465, 
    'max_depth': 14, 
    'num_leaves': 28, 
    'min_child_samples': 29, 
    'subsample': 0.5597689123597346, 
    'colsample_bytree': 0.6601202363535343, 
    'reg_alpha': 0.20732364284443197, 
    'reg_lambda': 0.004223724135505332,
    'objective': 'regression',
    'device': 'gpu',
    'metric': 'mape',
    'random_state': 42
    }


from xgboost import XGBRegressor

def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'device': 'cuda',
        'eval_metric': 'mape',
        'random_state': 42,
        'verbosity': -1
    }

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_val_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    return mape

# Optimize
#study_xgb = optuna.create_study(direction='minimize')
#study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

# Print best parameters
#best_params_xgb = study_xgb.best_params
#print("Best XGBoost Parameters:", best_params_xgb)
#print(f"Best MAPE: {study_xgb.best_value:.4f}")

best_params_xgb = {
    'n_estimators': 3231, 
    'learning_rate': 0.05895359669164567, 
    'max_depth': 7, 
    'min_child_weight': 4, 
    'subsample': 0.8319649088461181, 
    'colsample_bytree': 0.7107337151097438, 
    'gamma': 0.0019772108405958213, 
    'reg_alpha': 0.5384785820890761, 
    'reg_lambda': 0.7912823880613118,
    'device': 'cuda',
    'eval_metric': 'mape',
    'random_state': 42
}


from catboost import CatBoostRegressor

def objective_cat(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
        'loss_function': 'MAPE',
        'eval_metric': 'MAPE',
        'random_state': 42
    }

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train CatBoost
    model = CatBoostRegressor(silent=True, **param)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_val_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    return mape

# Optimize
#study_cat = optuna.create_study(direction='minimize')
#study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

# Print best parameters
#best_params_cat = study_cat.best_params
#print("Best CatBoost Parameters:", best_params_cat)
#print(f"Best MAPE: {study_cat.best_value:.4f}")

best_params_cat = {
    'n_estimators': 1891, 
    'learning_rate': 0.06761514972690001, 
    'depth': 8, 
    'min_data_in_leaf': 54,
    'l2_leaf_reg': 5.567375613813537, 
    'bagging_temperature': 0.15478395184586632, 
    'random_strength': 0.9462614107298501,
    'loss_function': 'MAPE',
    'eval_metric': 'MAPE',
    'random_state': 42
}


%%time

# Define models
xgb_model = XGBRegressor(**best_params_xgb)
lgb_model = LGBMRegressor(**best_params_lgb)
cat_model = CatBoostRegressor(silent=True,**best_params_cat)

meta_model = BayesianRidge()

# Create the stacking model
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=meta_model,
    n_jobs=-1
)

# Time the training process
stacking_model.fit(X, y)
predictions = stacking_model.predict(test_df)


predictions = np.round(np.expm1(predictions))
print(predictions)


submission = pd.DataFrame({'id': id_col, 'num_sold': predictions})
submission['num_sold'] = submission['num_sold'].clip(lower=0)
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

display(submission.head())

