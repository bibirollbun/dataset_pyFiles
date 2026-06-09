import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col='id')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col='id')


train.head()


def decompose(train, c, ax):
    df = train.groupby(['date',c])[['num_sold']].sum().reset_index().join(
                            train.groupby('date')[['num_sold']].sum(), on='date',rsuffix='_global')
    df['fractions'] = df['num_sold']/df['num_sold_global']
    for m in np.sort(df[c].unique()):
        mask = df[c]==m
        ax.plot(df[mask]['date'],df[mask]['fractions'],label=m)
    ax.legend(bbox_to_anchor=(1, 1))  


_, ax = plt.subplots()
decompose(train, 'product', ax)
plt.show()


import requests

def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

df = train[['date', 'country']].copy()
alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
df['alpha3'] = df['country'].map(dict(zip(
    np.sort(df['country'].unique()), alpha3s)))
years = np.sort(df['date'].dt.year.unique())
df['year'] = df['date'].dt.year
gdp = np.array([
    [get_gdp_per_capita(alpha3, year) for year in years]
    for alpha3 in alpha3s
])
gdp = pd.DataFrame(gdp/gdp.sum(axis=0), index=alpha3s, columns=years)
df['GDP'] = df.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)

_, ax = plt.subplots(figsize=(8,10))
decompose(train, 'country', ax)
for country in df['country'].unique():
    mask = df['country']==country
    ax.plot(df[mask]['date'],df[mask]['GDP'],'k--')
plt.show()



# Statistical distribution of class features
categorical_cols = ['country', 'store', 'product']

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette="viridis")
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()



# The mean of class features and targets
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=train, x=col, y='num_sold', estimator=np.mean, errorbar=None, palette="magma")
    plt.title(f'{col} vs Average Target')
    plt.xlabel(col)
    plt.ylabel('Average num_sold')
    plt.xticks(rotation=45)
    plt.show()


# Delete rows containing NaN in the num_sold column
# train = train.dropna(subset=['num_sold'])


import pandas as pd

from sklearn.impute import KNNImputer


def knn_impute(data, n_neighbors=5):

    """

    Impute missing values using KNN imputer.


    Parameters:

    - data: pd.DataFrame

        The input DataFrame with missing values.

    - n_neighbors: int, optional (default=5)

        The number of neighbors to use for imputation.


    Returns:

    - pd.DataFrame

        The DataFrame with missing values imputed.

    """

    # Initialize the KNN Imputer

    imputer = KNNImputer(n_neighbors=n_neighbors)


    # Fit and transform the data

    imputed_data = imputer.fit_transform(data)


    # Convert the result back to a DataFrame

    imputed_df = pd.DataFrame(imputed_data, columns=data.columns)


    return imputed_df

train = knn_impute(train, 5)


def process_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    
    df['YearSin'] = np.sin(2 * np.pi * df['Year'] / 4)
    df['YearCos'] = np.cos(2 * np.pi * df['Year'] / 4)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DaySin'] = np.sin(2 * np.pi * df['Day'] / 30)
    df['DayCos'] = np.cos(2 * np.pi * df['Day'] / 30)
    
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                                'Spring' if x in [3, 4, 5] else
                                                'Summer' if x in [6, 7, 8] else
                                                'Autumn')
    df = pd.get_dummies(df, columns=['Season'], prefix=['Season'])
    df = df.drop(date_col, axis=1)
    
    return df

train = process_date_features(train, 'date')
test = process_date_features(test, 'date')


train.columns


cat_cols = ['country', 'store', 'product', 'Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']

def one_hot_encode(df, columns):
    encoded_df = pd.get_dummies(df, columns=columns, drop_first=True)
    return encoded_df

train = one_hot_encode(train, cat_cols)
test = one_hot_encode(test, cat_cols)

test = test.reindex(columns=train.columns, fill_value=0)


logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])
test = test[X.columns]


from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the model
lgbm_regressor = LGBMRegressor()

# Define the parameter distribution
param_dist = {
    'num_leaves': np.arange(20, 150, 10),  # Range of leaves
    'max_depth': [-1, 5, 10, 15, 20],  # Depth of trees
    'learning_rate': np.logspace(-3, 0, num=10),  # Learning rates from 0.001 to 1
    'n_estimators': [100, 200, 300],  # Number of trees
    'boosting_type': ['gbdt', 'dart'],  # Boosting types
    'objective': ['regression', 'huber'],  # Objectives for regression
    'metric': ['l2', 'mae'],  # Metrics for evaluation
    'min_data_in_leaf': [20, 50, 100],  # Minimum data in leaf
    'lambda_l1': [0, 0.1, 1],  # L1 regularization
    'lambda_l2': [0, 0.1, 1]   # L2 regularization
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=lgbm_regressor, 
                                   param_distributions=param_dist, 
                                   n_iter=100,  # Number of parameter settings to sample
                                   scoring='neg_mean_squared_error',  # Scoring metric
                                   cv=5,  # Cross-validation
                                   verbose=1, 
                                   random_state=42,  # For reproducibility
                                   n_jobs=-1)  # Use all available cores

# Fit the model
random_search.fit(X, y)

# Best parameters
print("Best parameters found: ", random_search.best_params_)
print("Best score: ", -random_search.best_score_)  # Negate to get positive MSE


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def cross_val_lgbm_mape(X, y, test, groups, n_splits=5, **params):
    group_kf = GroupKFold(n_splits=n_splits)
    mape_scores = []
    preds = []

    for train_index, valid_index in group_kf.split(X, y, groups):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        preds.append(model.predict(test))

    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

model_params = {
    "objective": "regression",
    "metric": "rmse",
    "seed": 42,
    "verbose": -1,
}

groups = train['Year']

average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, groups, n_splits=5, **model_params)

print(f"Average MAPE across folds: {average_mape:.4f}")


test_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
lgb_preds_original = np.expm1(lgb_preds)
submission = pd.DataFrame({
    'id': test_submission['id'],
    'num_sold': lgb_preds_original
})
submission.to_csv('submission_lgb.csv', index=False)
print("Submission file created:")
print(submission.head())

