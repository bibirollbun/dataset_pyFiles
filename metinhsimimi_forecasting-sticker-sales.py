import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


sub=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df.head()


df.isna().sum(),test.isna().sum()


df.shape,test.shape


print(df['num_sold'].describe(),'\n',
df['num_sold'].mode(),'\n',
df['num_sold'].median())


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(x=df['num_sold'].dropna(), y=df['product'], ax=axes[0], palette='viridis')
axes[0].set_title('Bar Plot')

sns.violinplot(x=df['num_sold'].dropna(), ax=axes[1], color='orange')
axes[1].set_title('Violin Plot')

sns.histplot(df['num_sold'].dropna(), kde=True, ax=axes[2], color='purple', bins=10)
axes[2].set_title('Histogram with KDE')

plt.tight_layout()
plt.show()


for i in df.columns:
    print(i,':',df[i].unique())


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df['num_sold'] = imputer.fit_transform(df[['num_sold']])


def fix_dates(df, date_column='date'):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    
    df.drop(columns=[date_column], inplace=True)
    
    return df

df = fix_dates(df, date_column='date')
test= fix_dates(test, date_column='date')


df = pd.get_dummies(df, columns=['country', 'store', 'product'])
test = pd.get_dummies(test, columns=['country', 'store', 'product'])


df.head()


# Librarys for model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


x = df.drop(columns=['num_sold'],axis=1)
y = df['num_sold']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# models 
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
    "LightGBM": LGBMRegressor(random_state=42)
}


results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R²": r2, "MAE": mae, "Explained Variance": evs}


# Results 
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  R²: {metrics['R²']}")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  Explained Variance: {metrics['Explained Variance']}")
    print()


num_estimators = 10000
params = {
    'n_estimators': num_estimators,
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'max_depth': 8,
    'learning_rate': 0.03,
    'lambda_l1': 0.001,
    'lambda_l2': 0.01,
    'random_state': 42,
    'verbose': -1
}

lgbm = LGBMRegressor(**params)
lgbm.fit(x, y)

prediction = lgbm.predict(test)

sub['num_sold']=prediction
sub.to_csv('predictions_of_sale.csv', index=False)  # save !

# not good :(


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
rf.fit(x, y)

y_pred = rf.predict(test)

sub['num_sold']=y_pred
sub.to_csv('randomf.csv', index=False)  # save !

