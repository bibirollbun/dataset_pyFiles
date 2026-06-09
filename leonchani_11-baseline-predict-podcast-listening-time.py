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
from sklearn.linear_model import LinearRegression
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








# # deliting missing value rows
# df_train = df_train.dropna()
# df_test = df_test.dropna()


# def fillna_with_median(df: pd.DataFrame, columns_to_fill: list):
#     """
#     Imputes missing values in a given DataFrame column with the median of the respective column.

#     Args:
#         df (pd.DataFrame): The DataFrame to impute.
#         columns_to_fill (list): A list of column names to impute missing values.
#     """
#     for column in columns_to_fill:
#         if column in df.columns:
#             median_value = df[column].median()
#             df[column].fillna(median_value, inplace=True)
#         else:
#             print(f"warning: '{column}' does not exist")


# columns_to_fill_for_df_train = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
# columns_to_fill_for_df_test = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

# fillna_with_median(df_train, columns_to_fill_for_df_train)
# fillna_with_median(df_train, columns_to_fill_for_df_test)


# display(df_train[df_train['Episode_Length_minutes'].isnull()])
# df_train_null_rows_index = df_train[df_train['Episode_Length_minutes'].isnull()].index


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


# # deleting abnormal data
# df_train['minutes'] = df_train['Episode_Length_minutes'] - df_train['Listening_Time_minutes']
# df_train = df_train[df_train['minutes'] >= 0]

# del df_train['minutes']


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





# def one_hot_encode_columns(df, categorical_cols):
#     """
#     Performs one-hot encoding on categorical columns of the specified DataFrame.

#     Args:
#         df (pd.DataFrame): The DataFrame to process.
#         categorical_cols (list): A list of column names to perform One-Hot Encoding.

#     Returns:
#         pd.DataFrame: The DataFrame after one-hot encoding.
#     """
#     df_processed = df.copy()  # Create a copy to preserve the original DataFrame
#     for col in categorical_cols:
#         one_hot_encoded = pd.get_dummies(df_processed[col], prefix=col, dtype=int)
#         df_processed = pd.concat([df_processed, one_hot_encoded], axis=1)
#         df_processed = df_processed.drop(col, axis=1)
#     return df_processed


# # perform one_hot_encoding
# # categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns
# # categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day','Publication_Time', 'Episode_Sentiment']
# categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day','Publication_Time', 'Episode_Sentiment']
# df_train = one_hot_encode_columns(df_train, categorical_cols)
# df_test = one_hot_encode_columns(df_test, categorical_cols)


# from sklearn.preprocessing import LabelEncoder

# def preprocess_dataframe(df):
#     """
#     For the specified DataFrame, extract the episode number, perform numeric conversion, label encoding, 
#     and delete the specified columns.

#     Args:
#         df (pd.DataFrame): The DataFrame to be processed.

#     Returns:
#         pd.DataFrame: The processed DataFrame.
#     """
#     # Extracting episode numbers and converting numbers
#     df['Episode_Number_str'] = df['Episode_Title'].str.split(' ').str[1]
#     df['Episode_Number'] = pd.to_numeric(df['Episode_Number_str'], errors='coerce')
#     del df['Episode_Number_str']
#     del df['Episode_Title']

#     # # Label Encoding
#     # label_encoder = LabelEncoder()
#     # columns_to_encode = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
#     # for col in columns_to_encode:
#     #     encoded_col_name = f'{col}_Encoded'
#     #     df[encoded_col_name] = label_encoder.fit_transform(df[col])
#     #     del df[col]
    
#     return df

# # To see the labels assigned to each category:
# # mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# # print("\nCategory and label correspondence:", mapping)


# df_train = preprocess_dataframe(df_train)
# df_test = preprocess_dataframe(df_test)


# # null flg
# df_train['null_flg_Episode_Length_minutes'] = df_train['Episode_Length_minutes'].apply(lambda x: 1 if pd.isnull(x) else 0)
# df_train['null_Guest_Popularity_percentage'] = df_train['Guest_Popularity_percentage'].apply(lambda x: 1 if pd.isnull(x) else 0)

# df_test['null_flg_Episode_Length_minutes'] = df_test['Episode_Length_minutes'].apply(lambda x: 1 if pd.isnull(x) else 0)
# df_test['null_Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].apply(lambda x: 1 if pd.isnull(x) else 0)


# def apply_log_transformations_with_offset(df: pd.DataFrame, column_name: str, offset: float = 1e-5):
#     """
#     For a given DataFrame column, add an offset value,
#     then apply the base 10 logarithm and base e logarithm and add the result to the DataFrame as a new column.

#     Args:
#         df (pd.DataFrame): The DataFrame to apply the logarithmic transformation to.
#         column_name (str): The name of the column to apply the logarithmic transformation to.
#         offset (float, optional): The offset to add (defaults to 1e-5).
#     """
#     log10_col_name = f'{column_name}_log10'
#     log_e_col_name = f'{column_name}_log_e'

#     # add offset value
#     df[column_name + '_offset'] = df[column_name] + offset

#     df[log10_col_name] = np.log10(df[column_name + '_offset'])
#     df[log_e_col_name] = np.log(df[column_name + '_offset'])

#     # Optionally, to remove columns with offsets that are no longer needed.
#     df = df.drop(columns=[column_name + '_offset'])

#     return df


# # Apply a logarithmic transformation (with offset)
# df_train = apply_log_transformations_with_offset(df_train, 'Episode_Length_minutes')
# df_train = apply_log_transformations_with_offset(df_train, 'Host_Popularity_percentage')
# df_train = apply_log_transformations_with_offset(df_train, 'Guest_Popularity_percentage')
# df_train = apply_log_transformations_with_offset(df_train, 'Number_of_Ads')

# df_test = apply_log_transformations_with_offset(df_test, 'Episode_Length_minutes')
# df_test = apply_log_transformations_with_offset(df_test, 'Host_Popularity_percentage')
# df_test = apply_log_transformations_with_offset(df_test, 'Guest_Popularity_percentage')
# df_test = apply_log_transformations_with_offset(df_test, 'Number_of_Ads')

# del df_train['Episode_Length_minutes']
# del df_train['Host_Popularity_percentage']
# del df_train['Guest_Popularity_percentage']
# del df_train['Number_of_Ads']

# del df_test['Episode_Length_minutes']
# del df_test['Host_Popularity_percentage']
# del df_test['Guest_Popularity_percentage']
# del df_test['Number_of_Ads']


# def calculate_categorical_growth_rate_no_count(df: pd.DataFrame, categorical_col: str):
#     """
    # Calculates growth rates (or lift) based on the frequency of occurrence of each category for the categorical columns in the given DataFrame, 
    # and merges them into the original DataFrame. # The original categorical columns and count information are removed.
    
    # Args:
    # df (pd.DataFrame): The target DataFrame.
    # categorical_col (str): The name of the categorical column for which you want to calculate the growth rate.
    
    # Returns:
    # pd.DataFrame: The DataFrame with the growth rates calculated and the original categorical columns and count information removed.

#     """
#     tmp = df[categorical_col].value_counts().reset_index()
#     tmp.columns = [categorical_col, f'count_{categorical_col}']  # Give count columns unique names
#     tmp['growth_rate_' + categorical_col] = tmp[f'count_{categorical_col}'] / tmp[f'count_{categorical_col}'].mean()

#     df = pd.merge(df, tmp[[categorical_col, 'growth_rate_' + categorical_col]], on=categorical_col, how='inner')
#     df = df.drop(columns=[categorical_col])  # Here we only remove categorical_col because the count column only exists in tmp

#     return df


# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Episode_Title')
# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Podcast_Name')
# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Genre')
# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Publication_Day')
# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Publication_Time')
# df_train = calculate_categorical_growth_rate_no_count(df_train.copy(), 'Episode_Sentiment')


# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Episode_Title')
# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Podcast_Name')
# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Genre')
# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Publication_Day')
# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Publication_Time')
# df_test = calculate_categorical_growth_rate_no_count(df_test.copy(), 'Episode_Sentiment')


# df_train['flg_high_Host_Popularity'] = df_train['Host_Popularity_percentage'].apply(lambda x: 1 if x >= 70 else 0)
# df_train['flg_high_Guest_Popularity_percentage'] = df_train['Guest_Popularity_percentage'].apply(lambda x: 1 if x >= 70 else 0)

# df_test['flg_high_Host_Popularity'] = df_test['Host_Popularity_percentage'].apply(lambda x: 1 if x >= 70 else 0)
# df_test['flg_high_Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].apply(lambda x: 1 if x >= 70 else 0)


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


# del df_train['Podcast_Name']
# del df_train['Episode_Title']
# del df_train['Genre']
# del df_train['Publication_Day']
# del df_train['Publication_Time']
# del df_train['Episode_Sentiment']

# del df_test['Podcast_Name']
# del df_test['Episode_Title']
# del df_test['Genre']
# del df_test['Publication_Day']
# del df_test['Publication_Time']
# del df_test['Episode_Sentiment']

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


# # Define a base model for regression tasks
# estimators = [
#     ('gbdt_shallow', GradientBoostingRegressor(max_depth=2, random_state=42)),             # GBDT with shallow decision tree depth
#     ('gbdt_medium', GradientBoostingRegressor(max_depth=5, random_state=42)),              # GBDT with medium depth decision tree
#     ('gbdt_deep', GradientBoostingRegressor(max_depth=10, random_state=42)),               # GBDT with deep decision tree
#     ('rf_shallow', RandomForestRegressor(max_depth=3, random_state=42)),                   # Random forest with shallow decision tree depth
#     ('rf_deep', RandomForestRegressor(max_depth=15, random_state=42)),                     # Random forest with deep decision trees
#     ('mlp_large', MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=300)), # A neural network with a large number of layers
#     ('mlp_small', MLPRegressor(hidden_layer_sizes=(10,), random_state=42, max_iter=300)),  # A neural network with a small number of layers
#     ('linear_again', LinearRegression())                                                   # One linear model (reusing linear regression)
# ]

# # StackingRegressor
# stacking_model = StackingRegressor(
#     estimators=estimators,
#     final_estimator=lgb.LGBMRegressor(random_state=42),
#     cv=5
# )

# # モデルのトレーニングと予測
# stacking_model.fit(X, y)
# stacking_pred = stacking_model.predict(X_test)


# Define a base model for regression tasks
estimators = [
    ('gbdt_shallow', lgb.LGBMRegressor(max_depth=2, random_state=42, verbose=0)),             # GBDT with shallow decision tree depth
    ('gbdt_medium', lgb.LGBMRegressor(max_depth=5, random_state=42, verbose=0)),              # GBDT with medium depth decision tree
    ('gbdt_deep', lgb.LGBMRegressor(max_depth=10, random_state=42, verbose=0)),               # GBDT with deep decision tree
    # ('rf_shallow', RandomForestRegressor(max_depth=3, random_state=42)),                   # Random forest with shallow decision tree depth
    # ('rf_deep', RandomForestRegressor(max_depth=15, random_state=42)),                     # Random forest with deep decision trees
    # ('mlp_large', MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=300)), # A neural network with a large number of layers
    # ('mlp_small', MLPRegressor(hidden_layer_sizes=(10,), random_state=42, max_iter=300)),  # A neural network with a small number of layers
    # ('linear_again', LinearRegression())                                                   # One linear model (reusing linear regression)
]

# StackingRegressor
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=lgb.LGBMRegressor(random_state=77, verbose=0),
    cv=5
)

# Training the model
stacking_model.fit(X, y)


test = df_test.drop(columns=['id'])
pred = stacking_model.predict(test)
pred





# SEED = 42
# NUM_SPLITS = 5


# X = df_train.drop(columns=["id", "Listening_Time_minutes"])
# y = df_train["Listening_Time_minutes"]


# # Hyperparameters of LightGBM (fixed values)
# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt', # Boosting type: Gradient Boosting Decision Tree
#     'learning_rate': 0.01, # Learning rate: A smaller rate slows down learning but can improve accuracy
#     'num_leaves': 30,  # Number of leaves in a decision tree: Set smaller to prevent overfitting
#     'max_depth': 8,    # Maximum depth of a decision tree: Set smaller to prevent overfitting
#     # 'feature_fraction': 0.7, # Fraction of features to sample: 0.7 means 70% of columns are used for learning
#     # 'bagging_fraction': 0.7, # Fraction of data to sample: 0.7 means 70% of data is used for learning
#     # 'bagging_freq': 1, # Frequency to apply bagging_fraction: 1 means apply at each iteration
#     'lambda_l1': 0.1, # L1 regularization term weight: Prevents overfitting
#     'lambda_l2': 0.1, # L2 regularization term weight: Prevents overfitting
#     # 'min_child_samples': 20, # Minimum number of data needed in a leaf: Too small can cause overfitting
#     # 'min_data_in_leaf': 40, # Minimum number of data needed in a leaf: Too small can cause overfitting
#     # 'min_sum_hessian_in_leaf': 1e-2, # Minimum sum of hessians needed in a leaf: Too small can cause overfitting
#     'verbosity': -1, # Level of logging output during learning: -1 means no output
#     'early_stopping_rounds': 100, # Number of rounds for early stopping: Learning stops if the score doesn't improve for the specified rounds
#     'random_state': SEED, # Random seed: Ensures reproducibility
#     # 'colsample_bytree': 0.7, # Subsample ratio of columns when constructing each tree (prevents overfitting)
#     # 'min_child_weight': 1e-3, # Minimum sum of instance weight (hessian) needed in a child (leaf)
#     # 'path_smooth': 0.1, # Parameter to smooth decision tree path
#     'max_bin': 255, # Maximum number of bins to bucket feature values
#     'n_estimators': 100, # Number of boosting iterations (number of trees)
#     # 'scale_pos_weight': 1, # Weight for balancing unbalanced classes
#     # 'min_gain_to_split': 0.1, # Minimum loss reduction required to make a further partition on a leaf node
#     #'feature_fraction_bynode': 0.8, # Fraction of features to consider for each node split
#     #'force_col_wise': True, # Force column-wise histogram building
#     #'extra_trees': True, # Enable random forest like behavior
#     #'num_iterations': 2000, # Number of boosting iterations
#     # 'drop_rate': 0.1, # Dropout rate for Dropout Boosting
#     # 'skip_drop': 0.5, # Skip rate for Dropout Boosting
#     # 'top_rate': 0.2, # Ratio of top instances to keep for GOSS
#     # 'other_rate': 0.1, # Ratio of other instances to keep for GOSS
#     #'categorical_feature': [0, 1, 2], # Indices of categorical features (add as needed)
#     }

# # K-Fold Cross-Validation
# kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
# rmse_scores = []
# models = []  # A list to store trained models

# for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
#     print(f"Fold {fold+1}")
#     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#     model = lgb.LGBMRegressor(**params)
#     model.fit(X_train, y_train,
#               eval_set=[(X_val, y_val)],
#               eval_metric='rmse',
#               callbacks=[lgb.early_stopping(100, verbose=False)])

#     y_pred = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     rmse_scores.append(rmse)
#     models.append(model)  # Save the trained model

# # Show the mean and standard deviation of the cross-validation scores
# print("\nCross-validation RMSE scores:", rmse_scores)
# print(f"Mean RMSE: {np.mean(rmse_scores):.5f}")
# print(f"Standard Deviation of RMSE: {np.std(rmse_scores):.5f}")


# # Simple version of cross validation
# from sklearn.model_selection import cross_val_score

# # Preparing the LightGBM model (hyperparameters are defined in params)
# model = lgb.LGBMRegressor(**params)

# # Preparation for K-Fold cross-validation
# kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)

# # Evaluate RMSE using cross_val_score
# # By default, cross_val_score uses accuracy for classification and r2 for regression as the evaluation metric.
# # You need to explicitly specify negative_root_mean_squared_error in your scoring parameters and transform the results.
# neg_rmse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
# rmse_scores = -neg_rmse_scores

# print("Cross-validation RMSE scores:", rmse_scores)
# print(f"Mean RMSE: {np.mean(rmse_scores)}")


# # Visualize the distribution of RMSE scores with box plots
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=rmse_scores)
# plt.title('Distribution of RMSE Scores across Folds')
# plt.xlabel('Folds')
# plt.ylabel('RMSE Score')
# plt.grid(True)
# plt.show()





# # Get the importance of features for each model using gain
# lgb_importances = model.booster_.feature_importance(importance_type='gain')

# # Get the name of the feature
# feature_names = X.columns

# # Summarize feature importance in a DataFrame
# df_importances = pd.DataFrame({
#     'Feature': feature_names,
#     'LightGBM': lgb_importances,
# })

# # Visualize feature importance
# plt.figure(figsize=(12, 6))
# sns.barplot(x='LightGBM', y='Feature', data=df_importances.sort_values(by='LightGBM', ascending=False))
# plt.title('LightGBM Feature Importances (Gain)')
# plt.tight_layout()
# plt.show()

# display(df_importances.sort_values(by='LightGBM', ascending=False))





# import shap

# # LGBM SHAP values
# explainer_lgb = shap.TreeExplainer(model)
# shap_values_lgb = explainer_lgb.shap_values(X)


# shap.summary_plot(shap_values_lgb, X)


# # If shap_values_lgb is a list, convert it to a NumPy array
# if isinstance(shap_values_lgb, list):
#     shap_values_lgb = np.array(shap_values_lgb)

# # Handling the multiclass classification case
# if len(shap_values_lgb.shape) == 3:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=1).mean(axis=0)
# # Handling binary classification cases
# else:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=0)

# # Store in DataFrame
# df_importance = pd.DataFrame({
#     'feature': X.columns,
#     'shap_importance': shap_importance
# })

# # Sort by importance
# df_importance = df_importance.sort_values('shap_importance', ascending=False)

# # Show results
# display(df_importance)





# test_id = df_test["id"]
# test = df_test.drop(columns=['id'])
# submit_score = []

# for fold_, model in enumerate(models):
#     # predict test data
#     pred_ = model.predict(test)
#     submit_score.append(pred_)

# # predict test data
# pred = np.mean(submit_score, axis=0)





test_id = df_test["id"]

submission = pd.DataFrame({
    'id': test_id,
    'Listening_Time_minutes': pred
})

# Save
submission.to_csv('submission.csv', index=False)

submission




