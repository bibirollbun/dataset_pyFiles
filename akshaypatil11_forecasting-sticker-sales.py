import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
train.head()


train.shape


train.isnull().sum()


train['num_sold'] = train['num_sold'].fillna(train['num_sold'].median())


train.describe()


train.duplicated().sum()


train['num_sold'].hist()


def replace_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr
    median_value = df[column].median()
    df[column] = np.where((df[column]<lower)|(df[column]>upper), median_value, df[column])
    return df








test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test.head()


test.shape


test.isnull().sum()


test.describe()


concat_df = pd.concat([train, test], axis = 0)
concat_df.shape


numerical_columns = concat_df.select_dtypes(exclude='object').drop( 'id', axis = 1)
numerical_columns.columns


for col in numerical_columns.columns:
    concat_df = replace_outliers(concat_df, col)


categorical_columns = concat_df.select_dtypes(include='object').drop(['date'], axis = 1)
categorical_columns.columns



fig, axes = plt.subplots(1, 3, figsize= (15,10))
axes = axes.flatten()

for i, col in enumerate(categorical_columns.columns):
    sns.countplot(y=col, data=concat_df, ax = axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel('Distribution')

plt.tight_layout()
plt.show()


concat_df.date = pd.to_datetime(concat_df.date)
concat_df['year'] = concat_df.date.dt.year
concat_df['month'] = concat_df.date.dt.month
concat_df['day'] = concat_df.date.dt.day


for col in ['month', 'day']:
    max_val = 12 if col == 'month' else 31
    concat_df[f'{col}_sin'] = np.sin(2 * np.pi * concat_df[col]/max_val)
    concat_df[f'{col}_cos'] = np.cos(2 * np.pi * concat_df[col]/max_val)


concat_df['day_of_week'] = concat_df['date'].dt.dayofweek  
concat_df['is_weekend'] = (concat_df['date'].dt.weekday >= 5).astype(np.uint8)
concat_df['quarter'] = concat_df['date'].dt.quarter 


concat_df = concat_df.drop(['date', 'id'], axis=1)
concat_df = pd.get_dummies(concat_df)
concat_df


concat_df.shape


newtrain = concat_df.iloc[0:230130, :]
newtest = concat_df.iloc[230130:, :].drop('num_sold', axis = 1)


newtrain.head()


newtest.head()


newtrain.shape


newtest.shape


x = newtrain.drop('num_sold', axis = 1)
y = newtrain['num_sold']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(),
         'Bagging': BaggingRegressor(), 'Extra Tree': ExtraTreesRegressor(), 'LightGBM': LGBMRegressor(),
         'Gradient Boosting': GradientBoostingRegressor(), 'Adaboost': AdaBoostRegressor(),
         'XGB': XGBRegressor()}



def evaluate_models(x_train,x_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        predictions = model.fit(x_train, y_train).predict(x_test)
        accuracy = mean_absolute_percentage_error(y_test, predictions)
        results[name] = accuracy
    return results



results = evaluate_models(x_train, x_test, y_train, y_test, models)


best_model_name = min(results, key = results.get)
best_model = models[best_model_name]


print(f"best model is {best_model_name} with mape {results[best_model_name]}")


x_train = newtrain.drop('num_sold', axis = 1)
y_train = newtrain['num_sold']
x_test = newtest
y_pred = best_model.fit(x_train, y_train).predict(x_test)


solution = pd.DataFrame({'id':test['id'], 'num_sold': y_pred})
solution.head()


solution.to_csv('Solution.csv', index = False)




