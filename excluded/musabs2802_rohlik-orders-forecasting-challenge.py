import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor, BaggingRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/rohlik-orders-forecasting-challenge/train.csv')
df_train


df_train.info()


# Convert date to datetime type
df_train['date'] = pd.to_datetime(df_train['date'])


for col in df_train.columns:
    unqs = df_train[col].unique()
    print(col, unqs if len(unqs)<26 else '[...]', len(unqs))


df_train.duplicated().sum()


duplicates = df_train[df_train.duplicated(subset=['warehouse', 'date'], keep=False)]
duplicates.size


df_train.isna().sum()


df_train.loc[df_train['holiday_name'].isna(), 'holiday_name'] = 'No Holiday'
df_train.loc[df_train['precipitation'].isna(), 'precipitation'] = 0
df_train.loc[df_train['snow'].isna(), 'snow'] = 0


df_train_calendar = pd.read_csv('/kaggle/input/rohlik-orders-forecasting-challenge/train_calendar.csv')
df_train_calendar.head()


df_train_calendar.info()


# Convert date to datetime type
df_train_calendar['date'] = pd.to_datetime(df_train_calendar['date'])


df_train_calendar.columns


df_train.merge(df_train_calendar[['date', 'warehouse_limited']], on='date')['warehouse_limited'].describe()


display(pd.concat([df_train.groupby('warehouse')['orders'].sum(), (df_train.groupby('warehouse')['orders'].sum() / df_train['orders'].sum()).rename('proportion')], axis=1))

sns.barplot(df_train, x='warehouse', y='orders')
plt.show()


df_train['holiday_name'].unique()


plt.figure(figsize=(15, 7))

sns.lineplot(df_train, x='date', y='orders', hue='warehouse')

plt.show()


df_train.groupby('holiday')['orders'].mean()


(df_train.groupby('holiday')['orders'].mean()[1] - df_train.groupby('holiday')['orders'].mean()[0]) / df_train.groupby('holiday')['orders'].mean()[0]


plt.figure(figsize=(7, 7))
sns.barplot(df_train.groupby('holiday_name')['orders'].mean().sort_values().reset_index(), x='orders', y='holiday_name')

plt.axvline(df_train.groupby('holiday_name')['orders'].mean()['No Holiday'], color='red', linestyle='dotted', label='No Holiday')
plt.show()


df_train.head(3)


fig, ax = plt.subplots(2, 4, figsize=(12, 6))

sns.barplot(df_train.groupby('shutdown')['orders'].mean().reset_index(), x='shutdown', y='orders', ax=ax[0, 0])
sns.barplot(df_train.groupby('mini_shutdown')['orders'].mean().reset_index(), x='mini_shutdown', y='orders', ax=ax[0, 1])
sns.barplot(df_train.groupby('shops_closed')['orders'].mean().reset_index(), x='shops_closed', y='orders', ax=ax[0, 2])
sns.barplot(df_train.groupby('winter_school_holidays')['orders'].mean().reset_index(), x='winter_school_holidays', y='orders', ax=ax[0, 3])

sns.barplot(df_train.groupby('school_holidays')['orders'].mean().reset_index(), x='school_holidays', y='orders', ax=ax[1, 0])
sns.barplot(df_train.groupby('blackout')['orders'].mean().reset_index(), x='blackout', y='orders', ax=ax[1, 1])
sns.barplot(df_train.groupby('mov_change')['orders'].mean().reset_index(), x='mov_change', y='orders', ax=ax[1, 2])
sns.barplot(df_train.groupby('frankfurt_shutdown')['orders'].mean().reset_index(), x='frankfurt_shutdown', y='orders', ax=ax[1, 3])

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(4, 2, figsize=(15, 16))

sns.scatterplot(df_train, x='precipitation', y='orders', hue='warehouse', ax=ax[0, 0])
sns.kdeplot(df_train, x='precipitation', y='orders', hue='warehouse', ax=ax[0, 1])

sns.scatterplot(df_train, x='snow', y='orders', hue='warehouse', ax=ax[1, 0])
sns.kdeplot(df_train, x='snow', y='orders', hue='warehouse', ax=ax[1, 1])

sns.scatterplot(df_train, x='user_activity_1', y='orders', hue='warehouse', ax=ax[2, 0])
sns.kdeplot(df_train, x='user_activity_1', y='orders', hue='warehouse', ax=ax[2, 1])

sns.scatterplot(df_train, x='user_activity_2', y='orders', hue='warehouse', ax=ax[3, 0])
sns.kdeplot(df_train, x='user_activity_2', y='orders', hue='warehouse', ax=ax[3, 1])

plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(df_train.select_dtypes(exclude='object').corr(), annot=True, cmap='coolwarm')

plt.show()


def chi_square_test(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    results = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):  # Avoid duplicate pairs
            col1, col2 = categorical_cols[i], categorical_cols[j]
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append((col1, col2, chi2, p))
    
    results_df = pd.DataFrame(results, columns=["Feature 1", "Feature 2", "Chi2 Score", "p-value"])
    correlated_features = results_df[results_df["p-value"] < 0.05]
    
    return correlated_features

chi_square_results = chi_square_test(df_train.drop(columns=['orders'], axis=1))
print(chi_square_results)


df_copy = df_train.copy()
df_copy


df_copy['month'] = df_copy['date'].dt.month
df_copy['year'] = df_copy['date'].dt.year
df_copy['day'] = df_copy['date'].dt.day
df_copy['weekofyear'] = df_copy['date'].dt.dayofweek


# These columns are not available in test data
df_copy.drop(columns=['id', 'date', 'shutdown', 'mini_shutdown', 'blackout', 'mov_change', 'frankfurt_shutdown', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2'], inplace=True)


lencoder = LabelEncoder()
df_copy['warehouse'] = lencoder.fit_transform(df_copy['warehouse'])


# Encoding Categorical Features
ohencoder = OneHotEncoder(sparse=False)
holiday_encoded = ohencoder.fit_transform(df_copy[['holiday_name']])
encoded_df = pd.DataFrame(holiday_encoded, columns=ohencoder.get_feature_names_out(['holiday_name']))

df_copy = pd.concat([df_copy, encoded_df], axis=1)
df_copy = df_copy.drop('holiday_name', axis=1)

# Label Encoding for 'warehouse' column
lencoder = LabelEncoder()
df_copy['warehouse'] = lencoder.fit_transform(df_copy['warehouse'])


X_train, X_test, y_train, y_test = train_test_split(df_copy.drop('orders', axis=1), df_copy['orders'], test_size=0.2)

models = {
    'XGBRegressor': XGBRegressor(),
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
    'LGBMRegressor': LGBMRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'BaggingRegressor': BaggingRegressor(),
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'ExtraTreeRegressor': ExtraTreeRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor()
}

mape_results = []

for k, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape_results.append({'Model': k, 'MAPE': mape})

mape_df = pd.DataFrame(mape_results)
mape_df['MAPE'] = mape_df['MAPE'].apply(lambda x: f"{x:.5f}")
mape_df = mape_df.sort_values('MAPE')
mape_df


def plot_model_performance(mape_df):
    mape_df['MAPE'] = mape_df['MAPE'].astype(float)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='MAPE', data=mape_df)
    plt.title('Model Performance Comparison (MAPE)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_model_performance(mape_df)





df_test = pd.read_csv('/kaggle/input/rohlik-orders-forecasting-challenge/test.csv')
df_test


all_df = pd.concat([df_copy, df_test], sort=False)
all_df


df_test = all_df[all_df['orders'].isna()]
df_test


df_test['date'] = pd.to_datetime(df_test['date'])
# df_test.loc[df_test['holiday_name'].isna(), 'holiday_name'] = 'No Holiday'

df_test['month'] = df_test['date'].dt.month
df_test['year'] = df_test['date'].dt.year
df_test['day'] = df_test['date'].dt.day
df_test['weekofyear'] = df_test['date'].dt.dayofweek

df_test.drop(columns=['date', 'orders', 'holiday_name'], inplace=True)
df_test['warehouse'] = lencoder.fit_transform(df_test['warehouse'])

df_test.fillna(0, inplace=True)


y_pred = models[mape_df.iloc[0, 0]].predict(df_test.drop('id', axis=1))
y_pred


submission = pd.DataFrame({'id': df_test['id'], 'ORDERS': y_pred})
submission


submission.to_csv('submission.csv', index=False)




