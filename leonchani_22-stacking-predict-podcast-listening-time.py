# Table Manipulation, Calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Learning
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier, StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train


df_test








def fillna_by_group_mean(df: pd.DataFrame, group_columns: list, target_column: str):
    """
    Imputes missing values in a specified column with the mean for each specified group.

    Args:
        df (pd.DataFrame): DataFrame to impute.
        group_columns (list): List of column names to group by.
        target_column (str): Name of the column to impute missing values ​​to.
    """
    # Group by specified columns and calculate the mean of the target column
    tmp_mean = df.groupby(group_columns, as_index=False, dropna=False)[target_column].mean()
    mean_column_name = f'{target_column}_mean'
    tmp_mean = tmp_mean.rename(columns={target_column: mean_column_name})

    # Merge the mean values back into the original DataFrame
    df = pd.merge(df, tmp_mean, on=group_columns, how='left')

    # Replace missing values in the target column with the calculated mean
    df[target_column] = df[target_column].fillna(df[mean_column_name])

    # Delete the temporary mean column
    if mean_column_name in df.columns:
        del df[mean_column_name]

    return df


group_columns = ['Podcast_Name', 'Episode_Title']
column_Episode_Length_minutes = 'Episode_Length_minutes'
column_Guest_Popularity_percentage = 'Guest_Popularity_percentage'
column_Number_of_Ads = 'Number_of_Ads'

# df_train
df_train = fillna_by_group_mean(df_train.copy(), group_columns, column_Episode_Length_minutes)
df_train = fillna_by_group_mean(df_train.copy(), group_columns, column_Guest_Popularity_percentage)
df_train = fillna_by_group_mean(df_train.copy(), group_columns, column_Number_of_Ads)

# df_test
df_test = fillna_by_group_mean(df_test.copy(), group_columns, column_Episode_Length_minutes)
df_test = fillna_by_group_mean(df_test.copy(), group_columns, column_Guest_Popularity_percentage)


condition = df_train['Episode_Length_minutes'] <= df_train['Listening_Time_minutes']
df_train.loc[condition, 'Episode_Length_minutes'] = df_train.loc[condition, 'Listening_Time_minutes']


def clip_upper_to_quantile_keep_null(series, quantile=0.95):
    """
    Replaces values above the specified percentile with that percentile value and leaves null values alone.

    Args:
        series (pd.Series): The Series to process.
        quantile (float): The percentile to use as upper bound (range 0 to 1). Default is 0.95.

    Returns:
        pd.Series: A Series in which values outside the upper bound are replaced by the specified percentile value, leaving null values as is.
    """
    null_mask = series.isnull()        # Preserve the index of null values
    not_null_series = series.dropna()  # Series excluding null values

    if not not_null_series.empty:
        upper_bound = not_null_series.quantile(quantile)
        clipped_not_null_series = not_null_series.where(not_null_series <= upper_bound, upper_bound)
    else:
        clipped_not_null_series = pd.Series()  # If all original Series are null

    # Return null values to their original positions
    result_series = pd.Series(index=series.index)
    result_series[~null_mask] = clipped_not_null_series.reindex(series.index[~null_mask])
    result_series[null_mask] = np.nan

    return result_series


df_train['Episode_Length_minutes'] = clip_upper_to_quantile_keep_null(df_train['Episode_Length_minutes'], quantile=0.99)
df_train['Number_of_Ads'] = clip_upper_to_quantile_keep_null(df_train['Number_of_Ads'], quantile=0.99)

df_test['Episode_Length_minutes'] = clip_upper_to_quantile_keep_null(df_test['Episode_Length_minutes'], quantile=0.99)
df_test['Number_of_Ads'] = clip_upper_to_quantile_keep_null(df_test['Number_of_Ads'], quantile=0.99)





def target_encoding(df: pd.DataFrame, target_column: str, feature_column: str):
    """
    Performs target encoding on the specified feature column.

    Args:
        df (pd.DataFrame): The DataFrame to encode.
        target_column (str): The target variable (numeric) column name. Defaults to 'Listening_Time_minutes'.
        feature_column (str): The feature (categorical) column name to encode. Defaults to 'Podcast_Name'.

    Returns:
        pd.DataFrame: The DataFrame with the encoded feature columns added and the original feature columns removed.
    """
    # Create a dictionary of average target variables for each column
    encoding_dict = df.groupby([feature_column])[target_column].mean().to_dict()

    # Apply map to feature columns to convert categorical to numerical
    encoded_column_name = f'{feature_column}_numeric'
    df[encoded_column_name] = df[feature_column].map(encoding_dict)

    # Delete the original feature column
    # if feature_column in df.columns:
    #     del df[feature_column]

    return df


column_target = 'Listening_Time_minutes'

# Create a list of feature column names you want to encode.
features_to_encode = [
    'Podcast_Name',
    'Episode_Title',
    'Genre',
    'Publication_Day',
    'Publication_Time',
    'Episode_Sentiment'
]

for feature_column in features_to_encode:
    df_train = target_encoding(df_train, column_target, feature_column)

df_train


def map_category_to_numeric(df_train: pd.DataFrame, df_test: pd.DataFrame, category_col: str, numeric_col: str):
    """
    Using a dictionary of categories and values ​​created from the training data,
    it converts the specified categorical columns in the test data to their corresponding numeric values.

    Args:
        df_train (pd.DataFrame): Training data. Requires a categorical column and a numeric column.
        df_test (pd.DataFrame): Test data. Requires a categorical column.
        category_col (str): Name of the categorical column.
        numeric_col (str): Name of the corresponding numeric column.

    Returns:
        pd.DataFrame: the test data with a new column containing the numeric transformation
                      of the specified categorical column.
    """
    # Create a dictionary of category and numeric correspondence from the training data
    category_to_numeric_dict = df_train[[category_col, numeric_col]].drop_duplicates().set_index(category_col)[numeric_col].to_dict()

    # Map specified categorical columns in the test data to numeric values in a new column
    encoded_column_name = f'{category_col}_numeric'
    df_test[encoded_column_name] = df_test[category_col].map(category_to_numeric_dict)

    return df_test


Podcast_Name_name_col, Podcast_Name_numeric_col = 'Podcast_Name', 'Podcast_Name_numeric'
Episode_Title_col, Episode_Title_numeric_col = 'Episode_Title', 'Episode_Title_numeric'
Genre_col, Genre_numeric_col = 'Genre', 'Genre_numeric'
Publication_Day_col, Publication_Day_numeric_col = 'Publication_Day', 'Publication_Day_numeric'
Publication_Time_col, Publication_Time_numeric_col = 'Publication_Time', 'Publication_Time_numeric'
Episode_Sentiment_col, Episode_Sentiment_numeric_col = 'Episode_Sentiment', 'Episode_Sentiment_numeric'

map_category_to_numeric(df_train, df_test, Podcast_Name_name_col, Podcast_Name_numeric_col)
map_category_to_numeric(df_train, df_test, Episode_Title_col, Episode_Title_numeric_col)
map_category_to_numeric(df_train, df_test, Genre_col, Genre_numeric_col)
map_category_to_numeric(df_train, df_test, Publication_Day_col, Publication_Day_numeric_col)
map_category_to_numeric(df_train, df_test, Publication_Time_col, Publication_Time_numeric_col)
map_category_to_numeric(df_train, df_test, Episode_Sentiment_col, Episode_Sentiment_numeric_col)


deleting_columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in deleting_columns:
    if col in df_train.columns:
        del df_train[col]

for col in deleting_columns:
    if col in df_test.columns:
        del df_test[col]


def standardize_dataframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Standardizes the columns of the specified DataFrame and returns the updated original DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        cols (list[str]): A list of column names to standardize.
    
    Returns:
        pd.DataFrame: The DataFrame with the specified columns standardized (modifies the original DataFrame).

    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df[cols])
    scaled_values = scaler.transform(df[cols])
    df[cols] = scaled_values

    return df


# columns_to_standardize = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
columns_to_drop = ['id', 'Listening_Time_minutes']
columns_to_standardize = df_train.copy().drop(columns=columns_to_drop).columns

standardize_dataframe(df_train, columns_to_standardize)
standardize_dataframe(df_test, columns_to_standardize)





X = df_train.drop(columns=["id", "Listening_Time_minutes"])
y = df_train["Listening_Time_minutes"]


# Define a base model for regression tasks
estimators = [
    # ('gbdt_shallow', lgb.LGBMRegressor(max_depth=2, random_state=42, verbose=0)),             # GBDT with shallow decision tree depth
    # ('gbdt_medium', lgb.LGBMRegressor(max_depth=5, random_state=42, verbose=0)),              # GBDT with medium depth decision tree
    # ('gbdt_deep', lgb.LGBMRegressor(max_depth=10, random_state=42, verbose=0)),               # GBDT with deep decision tree
    # ('rf_shallow', RandomForestRegressor(max_depth=3, random_state=42)),                   # Random forest with shallow decision tree depth
    # ('rf_deep', RandomForestRegressor(max_depth=15, random_state=42)),                     # Random forest with deep decision trees
    # ('mlp_large', MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=300)), # A neural network with a large number of layers
    # ('mlp_small', MLPRegressor(hidden_layer_sizes=(10,), random_state=42, max_iter=300)),  # A neural network with a small number of layers

    ('lgbm_min', lgb.LGBMRegressor(max_depth=1, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),     # GBDT with min decision tree depth
    ('lgbm_shallow', lgb.LGBMRegressor(max_depth=2, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)), # GBDT with shallow decision tree depth
    ('lgbm_medium', lgb.LGBMRegressor(max_depth=5, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),  # GBDT with medium depth decision tree
    ('lgbm_deep', lgb.LGBMRegressor(max_depth=10, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),   # GBDT with deep decision tree
    ('lgbm_deeper', lgb.LGBMRegressor(max_depth=15, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),   # GBDT with deeper decision tree    
    # ('rf_shallow', RandomForestRegressor(max_depth=3, random_state=42, n_estimators=100, n_jobs=-1)),               # Random forest with shallow decision tree depth
    # ('rf_deep', RandomForestRegressor(max_depth=15, random_state=42, n_estimators=100, n_jobs=-1)),                # Random forest with deep decision trees
    # ('mlp_large', MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, random_state=42, max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)), # A neural network with a large number of layers
    # ('mlp_small', MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', alpha=0.0001, random_state=42, max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)),  # A neural network with a small number of layers
    ('ridge', Ridge(random_state=42)),
    ('lasso', Lasso(random_state=42)),
    ('linear_again', LinearRegression()),
    # ('svr_rbf', SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    # ('svr_linear', SVR(kernel='linear', C=1.0, epsilon=0.1)),
    # ('gbdt_min', GradientBoostingRegressor(max_depth=1, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),     # GBDT with min decision tree depth
    # ('gbdt_shallow', GradientBoostingRegressor(max_depth=2, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)), # GBDT with shallow decision tree depth
    # ('gbdt_medium', GradientBoostingRegressor(max_depth=5, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),  # GBDT with medium depth decision tree
    # ('gbdt_deep', GradientBoostingRegressor(max_depth=10, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),   # GBDT with deep decision tree
    # ('gbdt_deeper', GradientBoostingRegressor(max_depth=15, random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)),   # GBDT with deeper decision tree
]

# StackingRegressor
stacking_model = StackingRegressor(
    estimators=estimators,
    # final_estimator=lgb.LGBMRegressor(random_state=77, verbose=0),
    final_estimator=GradientBoostingRegressor(random_state=77, verbose=0),
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)

# Training the model
stacking_model.fit(X, y)





test = df_test.drop(columns=['id'])
pred = stacking_model.predict(test)
pred





test_id = df_test["id"]

submission = pd.DataFrame({
    'id': test_id,
    'Listening_Time_minutes': pred
})

# Save
submission.to_csv('submission.csv', index=False)

submission




