import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge 


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
gdp_per_capita = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')


train.head()


train.info()


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
    df['month'] = df[col].dt.month.astype('int')
    df['day'] = df[col].dt.day.astype('int')
    df['day_of_week'] = df[col].dt.dayofweek.astype('int')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('int')
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int')
    df['days_since_start'] = (df[col] - df[col].min()).dt.days
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


train_df_imputed = train_df_imputed.dropna()


train_df_imputed.head()


sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(train['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()


train_df_imputed['num_sold'] = np.log1p(train_df_imputed['num_sold'])


sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(train_df_imputed['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()


X = train_df_imputed.drop(columns=['num_sold'])
y = train_df_imputed['num_sold']


X_train, X_valid, y_train, y_valid  = train_test_split(X, y, test_size=0.1, random_state=42)


def tune_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-6, 1e-2),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
        'random_state': 42,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'mape'
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)


%%time
#study_xgb = optuna.create_study(direction='minimize')
#study_xgb.optimize(tune_xgb, n_trials=10)

#print("Best XGBoost params:", study_xgb.best_params)
#print("XGBoost MAPE:", study_xgb.best_value)


def tune_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
        'random_state': 42,
        'device': 'gpu',
        'metric': 'mape'
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)


%%time
#study_lgb = optuna.create_study(direction='minimize')
#study_lgb.optimize(tune_lgb, n_trials=10)

#print("Best LightGBM params:", study_lgb.best_params)
#print("LightGBM MAPE:", study_lgb.best_value)


def tune_catboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 15),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-6, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'MAPE',
        'eval_metric': 'MAPE',
        'random_state': 42
    }
    model = CatBoostRegressor(**params, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)


%%time
#study_catboost = optuna.create_study(direction='minimize')
#study_catboost.optimize(tune_catboost, n_trials=10)

#print("Best CatBoost params:", study_catboost.best_params)
#print("CatBoost MAPE:", study_catboost.best_value)


xgb_model = xgb.XGBRegressor(
    n_estimators=1523, learning_rate=0.01326, max_depth=9, min_child_weight=2,
    subsample=0.9756, colsample_bytree=0.6859, gamma=0.0012,
    reg_alpha=0.00134, reg_lambda=9.223, eval_metric='mape', tree_method='gpu_hist'
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=1963, learning_rate=0.07046, max_depth=14, min_child_samples=7,
    colsample_bytree=0.9881, subsample=0.9268, reg_alpha=0.1826,
    reg_lambda=0.00035, metric='mape', device='gpu'
)

catboost_model = CatBoostRegressor(
    n_estimators=2764, learning_rate=0.02683, depth=12,
    l2_leaf_reg=2.75e-06, bagging_temperature=0.278, random_strength=0.7234,
    loss_function='MAPE', task_type='GPU', devices='0', verbose=0
)


%%time
meta_model = LinearRegression()
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('catboost', catboost_model)],
    final_estimator=meta_model,
    n_jobs=1  
)


stacking_model.fit(X,y)


label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


test.head()


submission_ids = test['id']
predictions = stacking_model.predict(test)


predictions = np.expm1(predictions)


submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})


submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())

