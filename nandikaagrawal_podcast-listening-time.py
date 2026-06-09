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


!pip install pytorch-tabnet


# !pip install optuna


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head()


train_df.info()


train_df.describe()


def handle_missing_values(df, column_name, technique=0):
    if technique == 0:
        # Drop rows where the specified column has missing values
        df = df.dropna(subset=[column_name])
    
    elif technique == 1:
        # Fill missing values with a constant (e.g., 0 or 'Unknown')
        constant_value = 0  # You can change this to any constant or input argument
        df[column_name] = df[column_name].fillna(constant_value)
    
    elif technique == 2:
        # Fill missing values with the mean (only for numeric columns)
        if pd.api.types.is_numeric_dtype(df[column_name]):
            mean_value = df[column_name].mean()
            df[column_name] = df[column_name].fillna(mean_value)
        else:
            print(f"Warning: Column '{column_name}' is not numeric. Cannot apply mean imputation.")
    
    elif technique == 3:
        # Fill missing values with the median (only for numeric columns)
        if pd.api.types.is_numeric_dtype(df[column_name]):
            median_value = df[column_name].median()
            df[column_name] = df[column_name].fillna(median_value)
        else:
            print(f"Warning: Column '{column_name}' is not numeric. Cannot apply median imputation.")
    
    elif technique == 4:
        # Fill missing values with the mode (for categorical columns)
        mode_value = df[column_name].mode()[0]  # mode returns a series, so we take the first value
        df[column_name] = df[column_name].fillna(mode_value)
    
    else:
        print("Invalid technique specified. Please choose a valid technique (0-4).")
    
    return df


def handle_non_numerical_columns(df, column_name, technique=0):
    column = df[column_name]
    
    # Check the chosen technique and apply the corresponding encoding
    if technique == 0:
        # Label Encoding: Convert categories to integers
        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(column)
    
    elif technique == 1:
        # One-Hot Encoding: Convert categories to binary columns
        one_hot_encoded = pd.get_dummies(column, drop_first=True)
        one_hot_encoded = one_hot_encoded.astype(int)
        df = pd.concat([df, one_hot_encoded], axis=1)
        df.drop(column_name, axis=1, inplace=True)
    
    elif technique == 2:
        df[column_name] = df[column_name].str.extract(r'(\d+)', expand=False).astype(int) / 100
    
    elif technique == 3:
        # Frequency Encoding with Min-Max Scaling
        freq_map = column.value_counts() / len(df)  # Frequency encoding
        # Min-Max scaling of the frequency values
        scaler = MinMaxScaler()
        freq_scaled = pd.DataFrame(freq_map.values).to_numpy().reshape(-1, 1)  # Convert to NumPy array and reshape
        freq_scaled = scaler.fit_transform(freq_scaled)
        freq_scaled = freq_scaled.flatten()  # Convert it back to a 1D array
        
        # Map the scaled frequency values back to the column
        df[column_name] = column.map(dict(zip(freq_map.index, freq_scaled)))
    
    elif technique == 4:
        # Target Encoding: Replace categories by the mean of the target variable for each category
        mean_target_per_category = train_df.groupby(column_name)['Listening_Time_minutes']
        print(mean_target_per_category)
        train_df[column_name] = column.map(mean_target_per_category)
        test_df[column_name] = column.map(mean_target_per_category)
    
    else:
        raise ValueError("Invalid technique. Choose a technique between 0 and 4.")
    
    return df


numerical_features = train_df.select_dtypes(include=['number']).columns.tolist()
categorical_features = train_df.select_dtypes(exclude=['number']).columns.tolist()
print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


# Check for missing values in train dataset
print("Missing values in train dataset:")
print(train_df.isnull().sum())


train_df = handle_missing_values(train_df, 'Episode_Length_minutes', 3)
test_df = handle_missing_values(test_df, 'Episode_Length_minutes', 3)


train_df = handle_missing_values(train_df, 'Guest_Popularity_percentage', 3)
test_df = handle_missing_values(test_df, 'Guest_Popularity_percentage', 3)


train_df = handle_missing_values(train_df, 'Number_of_Ads', 3)
test_df = handle_missing_values(test_df, 'Number_of_Ads', 3)


def plot_numeric_distribution(df, column_name):
    """
    Plot the distribution of values in a numeric column using a histogram and a KDE plot.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the numeric column to plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Histogram
    sns.histplot(df[column_name], kde=True, bins=30)
    
    # Add title and labels
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Show the plot
    plt.show()


def plot_categorical_distribution(df, column_name):
    """
    Plot the distribution of values in a categorical column using a bar plot.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the categorical column to plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Bar Plot
    sns.countplot(x=df[column_name])
    
    # Add title and labels
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Show the plot
    plt.show()


plot_numeric_distribution(train_df, 'Episode_Length_minutes')


plot_numeric_distribution(train_df, 'Guest_Popularity_percentage')


plot_numeric_distribution(train_df, 'Number_of_Ads')


def constrain_column_values(df, column_name, min_value, max_value):
    df[column_name] = df[column_name].clip(lower=min_value, upper=max_value)
    
    return df


def plot_boxplot(df, column_name):
    """
    Plot a box plot for the specified column in the dataframe.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe containing the data.
    column_name : str
        The name of the numeric column to plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the box plot
    sns.boxplot(x=df[column_name])
    
    # Add title and labels
    plt.title(f'Box Plot of {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    
    # Show the plot
    plt.show()


plot_boxplot(train_df, 'Number_of_Ads')


plot_boxplot(train_df, 'Host_Popularity_percentage')


plot_boxplot(train_df, 'Guest_Popularity_percentage')


display(train_df['Number_of_Ads'].value_counts(dropna=False))


train_df = constrain_column_values(train_df, 'Number_of_Ads', 0, 3)
test_df = constrain_column_values(test_df, 'Number_of_Ads', 0, 3)


train_df['Number_of_Ads'] = train_df['Number_of_Ads'].astype(int)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].astype(int)


train_df = constrain_column_values(train_df, 'Guest_Popularity_percentage', 0, 100)
test_df = constrain_column_values(test_df, 'Guest_Popularity_percentage', 0, 100)


train_df = constrain_column_values(train_df, 'Host_Popularity_percentage', 0, 100)
test_df = constrain_column_values(test_df, 'Host_Popularity_percentage', 0, 100)


plot_boxplot(train_df, 'Episode_Length_minutes')


display(train_df['Episode_Length_minutes'].value_counts(dropna=False))


train_df['Episode_Length_minutes'].quantile(0.99)


train_df['Episode_Length_minutes'].quantile(0.999)


train_df['Episode_Length_minutes'].quantile(0.9999)


print(f"{(train_df['Episode_Length_minutes'] > 120).sum()=}")


train_df[train_df['Episode_Length_minutes'] > 120]


# optb = ContinuousOptimalBinning(
#     name="Episode_Length_minutes",
#     dtype="numerical",
#     prebinning_method="cart",  # Using CART for pre-binning
#     monotonic_trend="auto"    # Automatically determine monotonic trend
# )


# optb.fit(
#     train_df['Episode_Length_minutes'],
#     train_df['Listening_Time_minutes']
# )

# train_df['Episode_Length_minutes'] = pd.Series(
#     optb.transform(train_df['Episode_Length_minutes']),
#     index=train_df.index
# )


# optb.binning_table.build()
# print(optb.binning_table)
# print(type(train_df['Episode_Length_minutes'][0]))


# test_df['Episode_Length_minutes'] = pd.Series(
#     optb.transform(test_df['Episode_Length_minutes']),
#     index=test_df.index
# )


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap between all numerical columns in the dataframe.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe containing the data.
    """
    # Calculate the correlation matrix
    corr_matrix = df[numerical_features].corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, 
                linewidths=0.5, vmin=-1, vmax=1)
    
    # Add title
    plt.title('Correlation Heatmap', fontsize=16)
    
    # Display the plot
    plt.show()



plot_correlation_heatmap(train_df)


train_df.describe()


for col in categorical_features:
    plot_categorical_distribution(train_df, col)


train_df = handle_non_numerical_columns(train_df,'Genre', 1)
test_df = handle_non_numerical_columns(test_df,'Genre', 1)


day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
train_df['Publication_Day'] = train_df['Publication_Day'].map(day_mapping)
test_df['Publication_Day'] = test_df['Publication_Day'].map(day_mapping)


time_mapping = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}
train_df['Publication_Time'] = train_df['Publication_Time'].map(time_mapping)
test_df['Publication_Time'] = test_df['Publication_Time'].map(time_mapping)


sentiment_mapping = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2
}
train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_mapping)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].map(sentiment_mapping)


train_df.head()


train_df.info()


display(train_df['Episode_Title'].value_counts(dropna=False))


display(train_df['Podcast_Name'].value_counts(dropna=False))


train_df = handle_non_numerical_columns(train_df, 'Episode_Title', 2)
test_df = handle_non_numerical_columns(test_df, 'Episode_Title', 2)


# train_df = train_df.drop(columns=['Episode_Title'])
# test_df = test_df.drop(columns=['Episode_Title'])


train_df = handle_non_numerical_columns(train_df, 'Podcast_Name', 1)
test_df = handle_non_numerical_columns(test_df, 'Podcast_Name', 1)


train_df.min()


train_df.max()


train_df['Host_Popularity_percentage'] = train_df['Host_Popularity_percentage'] / 100
test_df['Host_Popularity_percentage'] = test_df['Host_Popularity_percentage'] / 100


train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'] / 100
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'] / 100


target_column = 'Listening_Time_minutes'
y = train_df[target_column] 
X = train_df.drop(columns=[target_column])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# target_column = 'Listening_Time_minutes'
# y = train_df[target_column].values  # y can be NumPy
# X = train_df.drop(columns=[target_column])  # X stays as DataFrame

# # kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # rmse_scores = []

# for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
#     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#     y_train, y_val = y[train_index], y[val_index]

#     # Drop 'id' column from training and validation sets
#     X_train = X_train.drop(columns=['id'])
#     X_val = X_val.drop(columns=['id'])

#     model = RandomForestRegressor(n_estimators=300, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_val)

#     mse = mean_squared_error(y_val, y_pred)
#     rmse = sqrt(mse)
#     rmse_scores.append(rmse)

#     print(f"Fold {fold} RMSE: {rmse:.4f}")

# average_rmse = np.mean(rmse_scores)
# print(f"\nAverage RMSE across all folds: {average_rmse:.4f}")


X_val = X_val.drop(columns=['id'])
X_train = X_train.drop(columns=['id'])


test_id = test_df['id']
test_df = test_df.drop(columns=['id'])


# def objective(trial):
#     param = {
#         'n_estimators': 5000,
#         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#         'random_state': 42,
#         'n_jobs': -1
#     }
    
#     model = LGBMRegressor(**param)
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='rmse',
#         early_stopping_rounds=100,
#         verbose=False
#     )
#     preds = model.predict(X_val)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     return rmse


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20)


# best_params = study.best_params
# best_params['n_estimators'] = 5000
# best_params['random_state'] = 42
# best_params['n_jobs'] = -1

# final_model = LGBMRegressor(**best_params)

# final_model.fit(
#     X_train, y_train,  
#     eval_metric='rmse'
# )
# preds = model.predict(X_val)
# rmse = mean_squared_error(y_val, preds, squared=False)


# rmse


X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()
X_test = test_df.to_numpy()


model = TabNetRegressor()

model.fit(
    X_train=X_train,
    y_train=y_train.reshape(-1, 1),
    eval_set=[(X_val, y_val.reshape(-1, 1))],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)



# model = LGBMRegressor()
# model.fit(X_train, y_train)


# model = RandomForestRegressor(n_estimators=300, random_state=42)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_val)

# mse = mean_squared_error(y_val, y_pred)  # Mean Squared Error
# rmse = sqrt(mse)  # Root Mean Squared Error

# print(f"RMSE: {rmse}")


# model = keras.Sequential([
#     layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(256, activation='elu'),
#     layers.Dense(1),
# ])


# model.compile(optimizer='adamw', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])
# history = model.fit(X_train, y_train, batch_size=256, epochs=150, verbose=0)
# history_df = pd.DataFrame(history.history)
# # Start the plot at epoch 5. You can change this to get a different view.
# history_df.loc[5:, ['loss']].plot();


# loss, rmse = model.evaluate(X_val, y_val)
# print("Test RMSE:", rmse)


# predictions = model.predict(test_df)


preds = model.predict(X_test).flatten()
submission["id"] = test_id
submission["listening_time"] = preds
submission.to_csv("tabnet_submission.csv", index=False)


# submission = pd.DataFrame({
#     "id": test_id,
#     "Listening_Time_minutes": predictions
# })

# submission.to_csv("submission.csv", index=False)
# print("Submission file created!")

