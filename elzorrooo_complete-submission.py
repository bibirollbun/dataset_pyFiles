# add any libraries that are not preinstalled in your envirnoment

# !pip install --upgrade category_encoders


# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
from category_encoders import TargetEncoder


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# metadata - filenames, global variables, etc... that determine the runnning configuration

# filenames 
df_train_path = '/kaggle/input/playground-series-s5e4/train.csv'
df_test_path = '/kaggle/input/playground-series-s5e4/test.csv'
df_samplesub_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'

df_output_directory = '/kaggle/working/'
df_output_path = '/kaggle/working/submission.csv'


df_train_cleaned_path = '/kaggle/input/cleaned-podcast-dataset/train_cleaned.csv'
df_test_cleaned_path = '/kaggle/input/cleaned-podcast-dataset/test_cleaned.csv'

# global variables
is_validation = False # true if we want to use Kaggle's 'test.csv' instead of the validation set
seed = 0


df_train = pd.read_csv(df_train_path)

if is_validation:
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=seed)
else:
    df_test = pd.read_csv(df_test_path)


# time cell for complexity analysis
start_time = time.time()


# Convert Episode_Title to numeric
df_train["Episode_Title"] = df_train["Episode_Title"].astype(str).str.extract(r"(\d+)").astype(int)
df_test["Episode_Title"] = df_test["Episode_Title"].astype(str).str.extract(r"(\d+)").astype(int)

df_train.head()


# Missing values check
print(df_train.isna().sum())


# Impute missing values
df_train["Number_of_Ads"].fillna(0, inplace=True)
df_train["Number_of_Ads"].head()


# Create function to impute missing values in various ways
def impute_missing_column(df, column, metric = 'mean', group_cols = None):
    """
    @brief Impute missing values in a specified column using a global or group-based strategy.

    @param df pandas.DataFrame: Input DataFrame.
    @param column str: Name of the column to impute.
    @param metric str: The imputation strategy to use. Must be 'mean' or 'median'. Default is 'mean'.
    @param group_cols list[str] or None: List of columns to group by for group-wise imputation.
                                          If None, the global metric is used.

    @return pandas.DataFrame: A new DataFrame with missing values imputed in the specified column.

    @throws ValueError: If an invalid metric is specified.
    """
    if metric not in ['mean', 'median']:
        raise ValueError("Metric must be 'mean' or 'median'")

    df_copy = df.copy()

    if group_cols is not None:
        if metric == 'mean':
            df_copy[column] = df_copy.groupby(group_cols)[column].transform(
                lambda x: x.fillna(x.mean()))
        else:  # median
            df_copy[column] = df_copy.groupby(group_cols)[column].transform(
                lambda x: x.fillna(x.median()))
    else:
        fill_value = df_copy[column].mean() if metric == 'mean' else df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(fill_value)

    return df_copy


# Episode_Length_minutes 
train_mean           = impute_missing_column(df_train, column="Episode_Length_minutes", metric="mean")
train_median         = impute_missing_column(df_train, column="Episode_Length_minutes", metric="median")

train_mean_podcast   = impute_missing_column(df_train, column="Episode_Length_minutes", metric="mean", group_cols=["Podcast_Name"])
train_median_podcast = impute_missing_column(df_train, column="Episode_Length_minutes", metric="median", group_cols=["Podcast_Name"])

train_mean_group     = impute_missing_column(df_train, column="Episode_Length_minutes", metric="mean", group_cols=["Podcast_Name", "Episode_Title"])
train_median_group   = impute_missing_column(df_train, column="Episode_Length_minutes", metric="median", group_cols=["Podcast_Name", "Episode_Title"])

# Guest_Popularity_percentage
train_mean           = impute_missing_column(train_mean, column="Episode_Length_minutes", metric="mean")
train_median         = impute_missing_column(train_median, column="Episode_Length_minutes", metric="median")

train_mean_podcast   = impute_missing_column(train_mean_podcast, column="Episode_Length_minutes", metric="mean", group_cols=["Podcast_Name"])
train_median_podcast = impute_missing_column(train_median_podcast, column="Episode_Length_minutes", metric="median", group_cols=["Podcast_Name"])

train_mean_group     = impute_missing_column(train_mean_group, column="Episode_Length_minutes", metric="mean", group_cols=["Podcast_Name", "Episode_Title"])
train_median_group   = impute_missing_column(train_median_group, column="Episode_Length_minutes", metric="median", group_cols=["Podcast_Name", "Episode_Title"])


print(train_mean.isna().sum())


# Categorical distributions
categorical_cols = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Number_of_Ads"]
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(x=train_mean[col].astype(str), ax=axs[i])
    axs[i].set_title(col)
    axs[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


# Bin Number_of_Ads
def bin_ads(x):
    if x in [0, 1, 2]:
        return str(int(x))
    else:
        return "3+"

# Binning popularity
def bin_popularity(x):
    if x < 25:
        return "[0,25)%"
    elif x < 50:
        return "[25,50)%"
    elif x < 75:
        return "[50,75)%"
    else:
        return "75+"


def enrich_train_df(df):
    """
    @brief Enrich the input DataFrame with new engineered features for modeling.
    
    This function adds binned variables, encoded features, dummy variables, and 
    custom engineered columns to enhance the dataset for modeling tasks.

    @param df pandas.DataFrame: Input DataFrame to be enriched. The function 
    modifies this DataFrame in-place without returning a copy.

    @return pandas.DataFrame: The same input DataFrame, enriched with additional features.
    """
    df["binned_num_of_adds"] = df["Number_of_Ads"].apply(bin_ads)

    df["trimmed_host_popularity"] = df["Host_Popularity_percentage"].clip(0, 100)
    df["trimmed_guest_popularity"] = df["Guest_Popularity_percentage"].clip(0, 100)

    df["host_popularity_binned"] = df["trimmed_host_popularity"].apply(bin_popularity)
    df["guest_popularity_binned"] = df["trimmed_guest_popularity"].apply(bin_popularity)

    cols_to_dummy = ["Podcast_Name", "Episode_Sentiment", "Publication_Day", 
                     "Publication_Time", "Genre", "binned_num_of_adds",
                     "host_popularity_binned", "guest_popularity_binned"]

    dummies = pd.get_dummies(df[["id"] + cols_to_dummy], columns = cols_to_dummy, drop_first = True)
    df = df.merge(dummies, on="id", how="left")

    df["sentiment_encoding"] = LabelEncoder().fit_transform(df["Episode_Sentiment"])
    df["adds_encoding"] = LabelEncoder().fit_transform(df["binned_num_of_adds"])
    df["host_popularity_encoding"] = LabelEncoder().fit_transform(df["host_popularity_binned"])
    df["guest_popularity_encoding"] = LabelEncoder().fit_transform(df["guest_popularity_binned"])

    df["is_weekend"] = df["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)
    df["has_guest"] = (~df["Guest_Popularity_percentage"].isna()).astype(int)

    df["total_popularity"] = np.where(
        df["Guest_Popularity_percentage"] > 0,
        df["Host_Popularity_percentage"] * df["Guest_Popularity_percentage"],
        df["Host_Popularity_percentage"] ** 2
    )

    return df



# Store DataFrames in a dictionary with their variable names as keys
train_variants = {
    "train_mean": train_mean,
    "train_median": train_median,
    "train_mean_podcast": train_mean_podcast,
    "train_median_podcast": train_median_podcast,
    "train_mean_group": train_mean_group,
    "train_median_group": train_median_group,
}



for name in train_variants:
    train_variants[name] = enrich_train_df(train_variants[name])



train_variants["train_mean"]


len(train_variants["train_median_podcast"].columns)


# time cell for complexity analysis
eda_execution_time = round(time.time() - start_time, 2)
print(f"Executed script in {eda_execution_time} seconds")


# time cell for complexity analysis
start_time = time.time()


df_train.head()


submission = df_test['id'].copy()

submission = pd.DataFrame({
    'id': df_test['id'],  
    'Listening_Time_minutes': df_train['Listening_Time_minutes'].median()      
})

submission


submission.to_csv(df_output_path, index = False)


# Submission: 70% of the Episode_Length, or 0 if Episode_Length is NaN
submission = pd.DataFrame({
    'id': df_test['id'],
    'Listening_Time_minutes': df_test['Episode_Length_minutes'].fillna(0) * 0.7
})



# time cell for complexity analysis
naive_execution_time = round(time.time() - start_time, 2)
print(f"Executed script in {naive_execution_time} seconds")


df_train = pd.read_csv(df_train_cleaned_path, index_col='id')

if is_validation:
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=seed)
else:
    df_test = pd.read_csv(df_test_cleaned_path, index_col='id')


df_train.dtypes.value_counts()


bool_cols = [col for col in df_train.columns if set(df_train[col].unique()) <= {0, 1}]
df_train[bool_cols] = df_train[bool_cols].astype(bool)



df_train.dtypes.value_counts()


X = df_train.drop(['Listening_Time_minutes'], axis=1)
y = df_train['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


linreg = LinearRegression()

linreg.fit(X_train, y_train)
lr_y_pred = linreg.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))

print("RMSE:", lr_rmse)


# alpha = {'alpha': list(range(50, 351))}
# ridge = Ridge()
# ridge_gscv = GridSearchCV(ridge, alpha, cv=5, scoring='neg_root_mean_squared_error')
# ridge_gscv.fit(X_train, y_train)

# r_alpha =ridge_gscv.best_params_['alpha']
# print("Best alpha:", r_alpha)
# print("RMSE:", -ridge_gscv.best_score_)


## Best alpha: 335


ridge = Ridge(alpha=335)

ridge.fit(X_train, y_train)
rlr_y_pred = ridge.predict(X_test)

rlr_rmse = np.sqrt(mean_squared_error(y_test, rlr_y_pred))

print("RMSE:", rlr_rmse)





rf = RandomForestRegressor(n_estimators=100, max_depth=10)

rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))

print("RMSE:", rf_rmse)


knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

knn_rmse = np.sqrt(mean_squared_error(y_test, knn_y_pred))

print("RMSE:", knn_rmse)



xgb = XGBRegressor()

xgb.fit(X_train, y_train)
xgb_y_pred = xgb.predict(X_test)

xgbrmse = np.sqrt(mean_squared_error(y_test, xgb_y_pred))

print("RMSE:", xgbrmse)



y_val_pred = xgb.predict(df_test)

xgbsubmission = pd.DataFrame({
    'id': df_test['id'],
    'Listening_Time_minutes': y_val_pred
})

xgbsubmission.to_csv(df_output_directory + 'baseline.csv', index=False)



start_time = time.time()


reg_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


def normalize_columns(df, columns_to_normalize):
    """
    Normalizes specified columns of a DataFrame to the range [0, 1].

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_normalize (list): List of column names to normalize.

    Returns:
        pd.DataFrame: A new DataFrame with normalized values in specified columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Apply scaler only to the specified columns
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df_normalized


def evaluate_datasets(mean_path, median_path, model):
    """
    /**
     * @brief Evaluate regression performance on mean and median imputed datasets with and without normalization.
     *
     * This function:
     * - Loads training datasets from the given file paths.
     * - Drops the 'id' column if it exists.
     * - Normalizes selected numerical columns to the range [0, 1].
     * - Trains and tests the provided regression model on:
     *     - Mean-imputed dataset
     *     - Normalized mean-imputed dataset
     *     - Median-imputed dataset
     *     - Normalized median-imputed dataset
     * - Computes the Root Mean Squared Error (RMSE) for each.
     *
     * @param mean_path Path to CSV file containing the mean-imputed training dataset.
     * @param median_path Path to CSV file containing the median-imputed training dataset.
     * @param model A regression model instance that supports fit(), predict(), and get_params().
     *
     * @return A dictionary with RMSE values for the different datasets:
     *     - "rmse_mean": RMSE on the mean-imputed dataset
     *     - "rmse_mean_norm": RMSE on the normalized mean-imputed dataset
     *     - "rmse_median": RMSE on the median-imputed dataset
     *     - "rmse_median_norm": RMSE on the normalized median-imputed dataset
     */
    """
    # Load datasets
    mean_train = pd.read_csv(mean_path)
    median_train = pd.read_csv(median_path)

    # Drop 'id' column if present
    for df in [mean_train, median_train]:
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)

    # Columns to normalize
    cols_to_normalize = [
        "Episode_Title", 
        "Episode_Length_minutes", 
        "trimmed_host_popularity", 
        "trimmed_guest_popularity", 
        "total_popularity"
    ]

    # Normalize
    mean_train_norm = normalize_columns(mean_train, cols_to_normalize)
    median_train_norm = normalize_columns(median_train, cols_to_normalize)

    # Prepare and evaluate Mean dataset
    X_mean = mean_train.drop(columns=['Listening_Time_minutes'])
    y_mean = mean_train['Listening_Time_minutes']
    X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean, y_mean, random_state=42)
    model
    model.fit(X_train_mean, y_train_mean, eval_set=[(X_test_mean, y_test_mean)], verbose=False)
    rmse_mean = np.sqrt(mean_squared_error(y_test_mean, model.predict(X_test_mean)))

    # Mean Normalized
    X_mean_norm = mean_train_norm.drop(columns=['Listening_Time_minutes'])
    y_mean_norm = mean_train_norm['Listening_Time_minutes']
    X_train_mean_norm, X_test_mean_norm, y_train_mean_norm, y_test_mean_norm = train_test_split(X_mean_norm, y_mean_norm, random_state=42)
    model.fit(X_train_mean_norm, y_train_mean_norm, eval_set=[(X_test_mean_norm, y_test_mean_norm)], verbose=False)
    rmse_mean_norm = np.sqrt(mean_squared_error(y_test_mean_norm, model.predict(X_test_mean_norm)))

    # Prepare and evaluate Median dataset
    X_median = median_train.drop(columns=['Listening_Time_minutes'])
    y_median = median_train['Listening_Time_minutes']
    X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(X_median, y_median, random_state=42)
    model.fit(X_train_median, y_train_median, eval_set=[(X_test_median, y_test_median)], verbose=False)
    rmse_median = np.sqrt(mean_squared_error(y_test_median, model.predict(X_test_median)))

    # Median Normalized
    X_median_norm = median_train_norm.drop(columns=['Listening_Time_minutes'])
    y_median_norm = median_train_norm['Listening_Time_minutes']
    X_train_median_norm, X_test_median_norm, y_train_median_norm, y_test_median_norm = train_test_split(X_median_norm, y_median_norm, random_state=42)
    model.fit(X_train_median_norm, y_train_median_norm, eval_set=[(X_test_median_norm, y_test_median_norm)], verbose=False)
    rmse_median_norm = np.sqrt(mean_squared_error(y_test_median_norm, model.predict(X_test_median_norm)))

    # Print Results
    print(f"Mean RMSE score: {rmse_mean:.4f}")
    print(f"Mean Normalized RMSE score: {rmse_mean_norm:.4f}")
    print()
    print(f"Median RMSE score: {rmse_median:.4f}")
    print(f"Median Normalized RMSE score: {rmse_median_norm:.4f}")

    return {
        'rmse_mean': rmse_mean,
        'rmse_mean_norm': rmse_mean_norm,
        'rmse_median': rmse_median,
        'rmse_median_norm': rmse_median_norm
    }


evaluate_datasets(
    df_output_directory + "train_cleaned_mean.csv",
    df_output_directory + "train_cleaned_median.csv",
    reg_xgb
)


podcast_result = evaluate_datasets(
    df_output_directory + 'train_cleaned_mean_pod.csv',
    df_output_directory + 'train_cleaned_median_pod.csv',
    reg_xgb
)


grouped_result = evaluate_datasets(
    df_output_directory + 'train_cleaned_mean_group.csv',
    df_output_directory + 'train_cleaned_median_group.csv',
    reg_xgb
)


imp_execution_time = round(time.time() - start_time, 2)
print(f"Executed script in {imp_execution_time} seconds")


start_time = time.time()


df_train = pd.read_csv(df_train_path, index_col='id')

if is_validation:
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=seed)
else:
    df_test = pd.read_csv(df_test_path, index_col='id')


def feature_eng(df):

    # map podcast names, genres, days, times, and sentiments to integers...
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    # ...and convert them to categorical types
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')


    # create a new column for the episode number as an integer category
    df['Episode_Number'] = df['Episode_Title'].str[8:].astype('category')
    
    df = df.drop(columns=['Episode_Title'])
    
    return df


df_train = feature_eng(df_train)
df_test = feature_eng(df_test)


df_train.head()


encode_columns = [
    'Podcast_Name', 'Episode_Number', 'Episode_Sentiment', 
    'Number_of_Ads', 'Publication_Day', 'Publication_Time'
]


encode_columns = [
    'Podcast_Name', 'Episode_Number', 'Episode_Sentiment', 
    'Number_of_Ads', 'Publication_Day', 'Publication_Time'
]

target_column = 'Listening_Time_minutes'


# Initialize category_encoders' TargetEncoder
encoder = TargetEncoder(
    cols=encode_columns,
    min_samples_leaf=1,
    smoothing=10,
    handle_missing='return_nan',
    handle_unknown='return_nan'
)

# Fit encoder on the training data
encoder.fit(df_train[encode_columns], df_train[target_column])

# Transform only the selected columns and add them to the original dataframes
df_train_encoded = encoder.transform(df_train[encode_columns])
df_test_encoded = encoder.transform(df_test[encode_columns])

# Rename the encoded columns to avoid overwriting original columns
df_train_encoded.columns = [f'{col}_encoded' for col in encode_columns]
df_test_encoded.columns = [f'{col}_encoded' for col in encode_columns]

# Add encoded columns to original dataframes
df_train = pd.concat([df_train, df_train_encoded], axis=1)
df_test = pd.concat([df_test, df_test_encoded], axis=1)

# Convert target to float32
df_train['Listening_Time_minutes'] = df_train['Listening_Time_minutes'].astype('float32')


df_train


df_train.columns


cat_columns = encode_columns


df_train[cat_columns] = df_train[cat_columns].astype('category')
y_train = df_train['Listening_Time_minutes']
df_train.drop(columns=['Listening_Time_minutes'], inplace=True)
X_train = df_train


model = lgb.LGBMRegressor(
    objective='l2',
    metric='rmse', 
    random_state=42,
    n_iter=12000,
    max_depth=15,
    learning_rate=0.008,
    num_leaves=480,
    colsample_bytree=0.25,
    verbosity=-1,
)


model.fit(X_train, y_train)


X_test = df_test
X_test[cat_columns] = df_test[cat_columns].astype('category')


df_subm = pd.read_csv(df_samplesub_path, index_col='id')
df_subm['Listening_Time_minutes'] = model.predict(X_test)
df_subm.to_csv(df_output_directory + 'submission-lgb.csv')


# time cell for complexity analysis
teb_execution_time = round(time.time() - start_time, 2)
print(f"Executed script in {teb_execution_time} seconds")

