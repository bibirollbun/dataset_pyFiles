!pip install --upgrade scikit-learn==1.3.1


print(__import__('sklearn').__version__)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


df_train.isna().sum()[df_train.isna().sum() > 0]


df_test.isna().sum()[df_test.isna().sum() > 0]


df_train.info()


def plot_histogram(df, column, bins=10, title=None, xlabel=None, ylabel='Frequency', figsize=(10, 6), color='skyblue', kde=False, print_mean_median=False):
    """
    plots a histogram for a specified column in a DataFrame using Seaborn.

    Args:
        df (pd.DataFrame): the DataFrame containing the data.
        column (str): the column to plot.
        bins (int): number of bins for the histogram (default is 10).
        title (str): title of the plot (default is None).
        xlabel (str): label for the x-axis (default is None).
        ylabel (str): label for the y-axis (default is 'Frequency').
        figsize (tuple): size of the figure (default is (10, 6)).
        color (str): color of the bars (default is 'skyblue').
        kde (bool): whether to overlay a Kernel Density Estimate (default is False).
        print_mean_median (bool): wheter to print mean and median (default is False)
    """

    if print_mean_median:
        print(f"mean: {df[column].mean():.2f}")
        print(f"median: {df[column].median():.2f}")
        
    sns.set_style('whitegrid')
    
    plt.figure(figsize=figsize)
    
    ax = sns.histplot(df[column], bins=bins, color=color, kde=kde, edgecolor='black', alpha=0.8)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    plt.show()


plot_histogram(df_train, 'Age', bins=20, title='Age Distribution', xlabel='Age', kde=True)


plot_histogram(df_train, 'Annual Income', bins=20, title='Annual Income Distribution', xlabel='Annual Income', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Number of Dependents', bins=20, title='Number of Dependents Distribution', xlabel='Number of Dependents', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Health Score', bins=20, title='Health Score Distribution', xlabel='Health Score', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Previous Claims', bins=20, title='Previous Claims Distribution', xlabel='Previous Claims', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Gender', bins=16, title='Gender Distribution', xlabel='Gender')


plot_histogram(df_train, 'Marital Status', bins=20, title='Marital Status Distribution', xlabel='Marital Status')


plot_histogram(df_train, 'Education Level', bins=20, title='Education Level Distribution', xlabel='Education Level')


plot_histogram(df_train, 'Occupation', bins=20, title='Occupation Distribution', xlabel='Occupation')


plot_histogram(df_train, 'Location', bins=20, title='Location Distribution', xlabel='Location')


plot_histogram(df_train, 'Policy Type', bins=20, title='Policy Type Distribution', xlabel='Policy Type')


plot_histogram(df_train, 'Customer Feedback', bins=20, title='Customer Feedback Distribution', xlabel='Customer Feedback')


plot_histogram(df_train, 'Smoking Status', bins=20, title='Smoking Status Distribution', xlabel='Smoking Status')


plot_histogram(df_train, 'Exercise Frequency', bins=20, title='Exercise Frequency Distribution', xlabel='Exercise Frequency')


plot_histogram(df_train, 'Property Type', bins=20, title='Property Type Distribution', xlabel='Property Type')


plot_histogram(df_train, 'Vehicle Age', bins=20, title='Vehicle Age Distribution', xlabel='Vehicle Age', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Credit Score', bins=20, title='Credit Score Distribution', xlabel='Credit Score', kde=True, print_mean_median=True)


plot_histogram(df_train, 'Insurance Duration', bins=20, title='Insurance Duration Distribution', xlabel='Insurance Duration Score', kde=True, print_mean_median=True)


df_train.head()


from time import perf_counter
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {perf_counter() - start:.2f} seconds")
        return result
    return wrapper


@timeit
def preprocess_data(
    df: pd.DataFrame,
    const_imputer: SimpleImputer,
    num_imputer: SimpleImputer,
    ordinal_encoder: OrdinalEncoder,
    target_encoder: TargetEncoder,
    is_test: bool = False
):
    """
    preprocess data

    1. impute missing data
    2. convert gender and smoking status to binary
    3. feature enginner
    4. enconding ordinal columns
    5. encoding category columns
    """
    if is_test: 
        X = df.drop(['id'], axis=1)
    else:
        X, y = df.drop(['id', 'Premium Amount'], axis=1), df['Premium Amount']
    
    # 1. impute missing data
    num_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
    if is_test:
        X[num_cols] = num_imputer.transform(X[num_cols])
    else:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    const_cols = ['Occupation', 'Customer Feedback', 'Marital Status']
    if is_test:
        X[const_cols] = const_imputer.transform(X[const_cols])
    else:
        X[const_cols] = const_imputer.fit_transform(X[const_cols])

    # 2. convert gender and smoking status to binary
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})
    X['Smoking Status'] = X['Smoking Status'].map({'No': 0, 'Yes': 1})

    # 3. feature enginner
    X['Policy Start Date'] = pd.to_datetime(X['Policy Start Date'], errors='coerce')
    X['Year'] = X['Policy Start Date'].dt.year
    X['Day'] = X['Policy Start Date'].dt.day
    X['Month'] = X['Policy Start Date'].dt.month
    X['Month_name'] = X['Policy Start Date'].dt.month_name()
    X['Day_of_week'] = X['Policy Start Date'].dt.day_name()
    X['Week'] = X['Policy Start Date'].dt.isocalendar().week
    X['Year_sin'] = np.sin(2 * np.pi * X['Year'])
    X['Year_cos'] = np.cos(2 * np.pi * X['Year'])
    X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
    X['Day_sin'] = np.sin(2 * np.pi * X['Day'] / 31)
    X['Day_cos'] = np.cos(2 * np.pi * X['Day'] / 31)
    
    X = X.drop(['Policy Start Date', 'Year', 'Day', 'Month'], axis=1)
    
    # 4. encoding ordinal columns
    if is_test:
        X[['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']] = ordinal_encoder.transform(
            X[['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']]
        )
    else:
        X[['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']] = ordinal_encoder.fit_transform(
            X[['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']]
        )

    # 5. encoding category columns
    category_cols = X.select_dtypes(include=['object']).columns
    if is_test:
        X[category_cols] = target_encoder.transform(X[category_cols])
    else:
        X[category_cols] = target_encoder.fit_transform(X[category_cols], y)

    if is_test:
        return X
        
    return X, y


const_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
num_imputer = SimpleImputer(strategy='median')


ordinal_map = {
    'Education Level': ["High School", "Bachelor's", "Master's", "PhD"],
    'Policy Type': ['Basic', 'Comprehensive', 'Premium'],
    'Customer Feedback': ['unknown', 'Poor', 'Average', 'Good'],
    'Exercise Frequency': ['Rarely', 'Monthly', 'Weekly', 'Daily'],
}

ordinal_encoder = OrdinalEncoder(categories=[ordinal_map[col] for col in ['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']])


target_encoder = TargetEncoder(smooth="auto", target_type='continuous', cv=5, random_state=42)


X, y = preprocess_data(df_train, const_imputer, num_imputer, ordinal_encoder, target_encoder)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"shape train: {X_train.shape}, val: {X_val.shape}")


lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")


# lgbm = LGBMRegressor(random_state=42)

# param_grid = {
#     'num_leaves': [31, 50, 70],
#     'max_depth': [10, 20, 30],
#     'learning_rate': [0.05, 0.1, 0.15],
#     'n_estimators': [50, 100, 200]
# }

# grid_search = GridSearchCV(
#     estimator=lgbm, 
#     param_grid=param_grid, 
#     cv=5, 
#     scoring='neg_mean_squared_error',
#     n_jobs=-1
# )

# grid_search.fit(X_train, y_train)
# hypertuned_model = grid_search.best_estimator_
# best_params = grid_search.best_params_

# y_pred = hypertuned_model.predict(X_val)

# mse = mean_squared_error(y_val, y_pred)
# rmse = np.sqrt(mse)

# print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")
# print("best paramaters:", best_params)
# print("=" * 40)


hypertuned_xgb = LGBMRegressor(
    learning_rate=0.05,
    max_depth=20,
    n_estimators=200,
    num_leaves=70,
    random_state=42
)

hypertuned_xgb.fit(X_train, y_train)

X = preprocess_data(df_test, const_imputer, num_imputer, ordinal_encoder, target_encoder, is_test=True)

y_pred = hypertuned_xgb.predict(X)

df_test['Premium Amount'] = y_pred
df_test[['id', 'Premium Amount']].to_csv('/kaggle/working/submission.csv', index=False)

