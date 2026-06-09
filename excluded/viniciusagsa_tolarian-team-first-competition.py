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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")


def num_hist_reg_plot(df, columns=None, y=None, bins=10 ):
    """
    Generates distribution plots (histograms with KDE) and binned trend plots 
    showing how the mean of a target variable behaves across different ranges 
    of numeric features.

    Parameters:
    -----------
    df : pandas.DataFrame  
        The dataset containing the features and target variable.

    columns : list of str or str  
        List of numeric feature names to be plotted.  
        Can be a single column name as a string.

    y : str  
        Name of the target variable to be analyzed across the binned feature values.

    bins : int, default=10  
        Number of bins to divide the numeric features into for aggregation.

    Returns:
    --------
    None  
        This function does not return anything. It generates and displays the plots.
    """
    df = df.copy()
    n_col = len(columns)
    fig, ax = plt.subplots(n_col, 2, figsize=(12, n_col * 4))

    # Ajust for 1 column
    if n_col == 1:
        ax = [ax]

    for i, feature in enumerate(columns):
        sns.histplot(
            data=df,
            bins=bins,
            x=feature,
            kde=True,
            ax=ax[i][0]
        )
        ax[i][0].set_title(f'Distribution of {feature}')
    
        # Create bins for each feature
        feature_bin = feature + '_bin'
        df[feature_bin] = pd.cut(df[feature], bins=bins)
        
        # Calculate the mean for each bin and get the center of each bin;
        df_grouped = df.groupby(feature_bin, observed=True)[y].mean().reset_index()
        df_grouped[feature_bin + '_center'] = df_grouped[feature_bin].apply(lambda x: x.mid)
        
        # Second Plot
        sns.lineplot(
            data=df_grouped,
            x = feature_bin + '_center',
            y = y,
            markers=True,
            ax=ax[i][1],
        )
        ax[i][1].set_title(f'Mean Listening_Time_minutes per {feature} (Binned)')
        
    plt.tight_layout()
    plt.show()


def cat_count_mean_plot(df, columns=None, y=None):
    """
    Generates bar plots showing the distribution of categorical variables and 
    line plots of the average of a target variable for each category.

    Parameters:
    -----------
    df : pandas.DataFrame  
        The dataset containing the features and the target variable.

    columns : list of str or str  
        List of categorical column names to be plotted.  
        Can be passed as a single column name (string).

    y : str  
        Name of the target variable to compute the average for each category.

    Returns:
    --------
    None  
        This function does not return any value. It displays the generated plots.
    """
    
    n_col = len(columns)
    fig, ax = plt.subplots(n_col, 2, figsize=(12, n_col * 4))

    # Ajust for 1 column
    if n_col == 1:
        ax = [ax]

    for i, feature in enumerate(columns):
        # Mean of Listening_Time para each category
        df_grouped = df.groupby(feature)['Listening_Time_minutes'].mean().reset_index()
        df_grouped = df_grouped.sort_values(by='Listening_Time_minutes')

        # First plot
        sns.countplot(
            data=df,
            x=feature,
            ax=ax[i][0],
            color='tab:blue',
            order=df_grouped[feature]
        )
        ax[i][0].set_title(f'Distribution of {feature}')
        ax[i][0].tick_params(axis='x', rotation=45)

        # Second plot
        sns.lineplot(
            data=df_grouped,
            x=feature,
            y='Listening_Time_minutes',
            markers=True,
            ax=ax[i][1],
        )
        ax[i][1].set_title(f'Mean Listening_Time_minutes per {feature}')
        ax[i][1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df.drop(columns='id', inplace=True)


df.head()


df.info()


df.nunique()


df.describe(percentiles=[.25, .5, .75, .95, .999])


df['Episode_Length_minutes'] = df['Episode_Length_minutes'].apply(lambda x: min(x, 120))
df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].apply(lambda x: min(x, 100))
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].apply(lambda x: min(x, 100))
df['Number_of_Ads'] = df['Number_of_Ads'].apply(lambda x: min(x, 3))


df.describe(percentiles=[.25, .5, .75, .95, .999])


numeric_col = df.select_dtypes(include=[np.number]).columns
category_col = df.select_dtypes(include=['object']).columns


num_hist_reg_plot(df, columns=numeric_col, y='Listening_Time_minutes', bins=10 )


cat_count_mean_plot(df, columns=category_col, y='Listening_Time_minutes')


# Create new columns for further analysis
df['Episode_Title_Number'] = df['Episode_Title'].str[-2:].astype('int')
df['Listening_Time_Percentage'] = df['Listening_Time_minutes'] / df['Episode_Length_minutes']



# num_hist_reg_plot(df, columns=['Episode_Title_Number', 'Listening_Time_Percentage'], y='Listening_Time_minutes', bins=10 )


plt.figure(figsize=(10,5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


df.head()


X = df.drop(columns = "Listening_Time_minutes")
y = df["Listening_Time_minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# List columns for each type of pipeline
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Title_Number']
categorical_cols_1hot = ['Podcast_Name', 'Genre']
categorical_cols_ordinal = ['Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Ordered values for ordinal imputer
map_Publication_Day = [['Sunday', 'Thursday', 'Friday', 'Saturday', 'Wednesday', 'Monday', 'Tuesday']]
map_Publication_Time = [['Evening', 'Morning', 'Afternoon', 'Night']]
map_Episode_Sentiment = [['Negative', 'Neutral', 'Positive']]


# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# One-hot encoding for nominal categorical variables
categorical_1hot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Ordinal encoding for ordered categorical variables
ordinal_transformer_Pub_Day = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=map_Publication_Day))
])

ordinal_transformer_Pub_Time = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=map_Publication_Time))
])

ordinal_transformer_Epi_Sentiment = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=map_Episode_Sentiment))
])

# Column Transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat1hot', categorical_1hot_transformer, categorical_cols_1hot),
    ('Pub_Day_ord', ordinal_transformer_Pub_Day, [categorical_cols_ordinal[0]]),
    ('Pub_Time_ord', ordinal_transformer_Pub_Time, [categorical_cols_ordinal[1]]),
    ('Epi_Sentiment_ord', ordinal_transformer_Epi_Sentiment, [categorical_cols_ordinal[2]])
])


# Linear Regressor 
lr_model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# Fit and predict
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# # Random Forest Regressor
# rf_model = Pipeline(steps=[
#     ('preprocessing', preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42))
# ])

# # Hyperparameter grid
# param_grid = {
#     'regressor__n_estimators': [50, 100, 200],
#     'regressor__max_depth': [None, 5, 10],
#     'regressor__min_samples_split': [2, 5, 10]
# }

# # Grid search with cross-validation
# grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)


# # Fit grid search
# grid_search.fit(X_train, y_train)

# # Best model
# best_model = grid_search.best_estimator_

# # Predictions and evaluation
# y_pred = best_model.predict(X_test)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)

# # Output
# print("Best Parameters:", grid_search.best_params_)
# print(f"Test RMSE: {rmse:.2f}")
# print(f"Test R²: {r2:.2f}")


test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_id = test["id"].copy()

test['Episode_Title_Number'] = test['Episode_Title'].str[-2:].astype('int')

test_preds = lr_model.predict(test)
submission_df = pd.DataFrame({
    "id": test_id,
    "Listening_Time_minutes": test_preds
})

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())




