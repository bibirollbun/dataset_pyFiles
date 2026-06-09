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


!pip install tensorflow


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import train and test data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.info()
train.head()


test.info()
test.head()


print(train['Listening_Time_minutes'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(train['Listening_Time_minutes'], color='g', bins=100, hist_kws={'alpha': 0.4})


# Set id as index
train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


train_num = train.select_dtypes(include = ['float64', 'int64'])
train_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


train_num.describe()


# Episode Length Minutes
plt.scatter(train['Episode_Length_minutes'], train['Listening_Time_minutes'])
plt.title('Episode Length vs Listening Time')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()

filter_ep_length=train.loc[train['Episode_Length_minutes']>150,:]
filter_ep_length


# Episode_Length_minutes
Q1 = train['Episode_Length_minutes'].quantile(0.25)
Q3 = train['Episode_Length_minutes'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print('Lower bound:',lower_bound)
print('Upper bound:',upper_bound)

outlier_mask = (train['Episode_Length_minutes'] < lower_bound) | (train['Episode_Length_minutes'] > upper_bound)
outlier_rows = train[outlier_mask]
train_cleaned=train[~outlier_mask].copy()
outlier_rows


# Number_of_Ads 
Q1 = train['Number_of_Ads'].quantile(0.25)
Q3 = train['Number_of_Ads'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print('Lower bound:',lower_bound)
print('Upper bound:',upper_bound)

outlier_mask = (train['Number_of_Ads'] < lower_bound) | (train['Number_of_Ads'] > upper_bound)
outlier_rows = train[outlier_mask]
train_cleaned=train[~outlier_mask].copy()
outlier_rows


test_cleaned=test.copy()
test_cleaned['Episode_Length_minutes'] = test_cleaned['Episode_Length_minutes'].clip(upper=181.58)
test_cleaned['Episode_Length_minutes'].describe()


test_cleaned['Number_of_Ads'] = test_cleaned['Number_of_Ads'].clip(upper=5)
test_cleaned['Number_of_Ads'].describe()


# Number_of_Ads
train_cleaned['Number_of_Ads']=train_cleaned['Number_of_Ads'].fillna(train_cleaned['Number_of_Ads'].median())


from sklearn.preprocessing import LabelEncoder

# Create copies
train_cleaned_encoded = train_cleaned.copy()
test_cleaned_encoded = test_cleaned.copy()

# Label encode Episode_Title
le_ti = LabelEncoder()
train_cleaned_encoded['Episode_Title_LE'] = le_ti.fit_transform(train_cleaned_encoded['Episode_Title'])
test_cleaned_encoded['Episode_Title_LE'] = le_ti.transform(test_cleaned_encoded['Episode_Title'])  # Only transform

# Label encode Episode_Sentiment
le_sen = LabelEncoder()
train_cleaned_encoded['Episode_Sentiment_LE'] = le_sen.fit_transform(train_cleaned_encoded['Episode_Sentiment'])
test_cleaned_encoded['Episode_Sentiment_LE'] = le_sen.transform(test_cleaned_encoded['Episode_Sentiment'])  # Only transform

# Check the classes
print(list(le_ti.classes_))
print(list(le_sen.classes_))



train_cleaned_encoded = train_cleaned_encoded.drop(['Episode_Sentiment', 'Episode_Title'], axis=1)


train_cleaned_encoded=train_cleaned_encoded.dropna()
test_cleaned_encoded['Guest_Popularity_percentage']=test_cleaned_encoded['Guest_Popularity_percentage'].fillna(train_cleaned_encoded['Guest_Popularity_percentage'].median())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Categorical columns for one-hot encoding
categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

# Features and target
X_train = train_cleaned_encoded.drop(columns=['Episode_Length_minutes', 'Listening_Time_minutes'])
y_train = train_cleaned_encoded['Episode_Length_minutes']

X_test = test_cleaned_encoded.drop(columns=['Episode_Length_minutes'])
y_test = test_cleaned_encoded['Episode_Length_minutes']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

# Build pipeline using LinearRegression instead of RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict missing episode lengths
test_missing_episode_length = test_cleaned_encoded[test_cleaned_encoded['Episode_Length_minutes'].isnull()]
X_test_missing = test_missing_episode_length.drop(columns=['Episode_Length_minutes'])
predicted_episode_lengths = pipeline.predict(X_test_missing)

# Fill missing values
test_cleaned_encoded.loc[
    test_cleaned_encoded['Episode_Length_minutes'].isnull(),
    'Episode_Length_minutes'
] = predicted_episode_lengths

# Evaluate model on non-null data
y_test_non_null = y_test.dropna()
X_test_non_null = X_test.loc[y_test_non_null.index]
y_pred_non_null = pipeline.predict(X_test_non_null)

# Metrics
mae = mean_absolute_error(y_test_non_null, y_pred_non_null)
rmse = np.sqrt(mean_squared_error(y_test_non_null, y_pred_non_null))
r2 = r2_score(y_test_non_null, y_pred_non_null)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# # Categorical columns for one-hot encoding
# categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

# # Features (X) and Target (y) for training
# X_train = train_cleaned_encoded.drop(columns=['Episode_Length_minutes', 'Listening_Time_minutes'])  # Drop the target variable
# y_train = train_cleaned_encoded['Episode_Length_minutes']  # The target variable

# X_test = test_cleaned_encoded.drop(columns=['Episode_Length_minutes'])  # Drop the target variable for test set
# y_test = test_cleaned_encoded['Episode_Length_minutes']  # The target variable for the test set

# # Define preprocessor for categorical columns
# preprocessor = ColumnTransformer(
#     transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
#     remainder='passthrough'
# )

# # Build the RandomForest pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # You can tune the n_estimators and other parameters
# ])

# # Fit the model on the training data
# pipeline.fit(X_train, y_train)

# # Identify rows in the test set where Episode_Length_minutes is missing
# test_missing_episode_length = test_cleaned_encoded[test_cleaned_encoded['Episode_Length_minutes'].isnull()]

# # Extract the features for rows with missing 'Episode_Length_minutes'
# X_test_missing = test_missing_episode_length.drop(columns=['Episode_Length_minutes'])

# # Predict the missing episode lengths
# predicted_episode_lengths = pipeline.predict(X_test_missing)

# # Fill the missing 'Episode_Length_minutes' values in the original test set with the predicted values
# test_cleaned_encoded.loc[test_cleaned_encoded['Episode_Length_minutes'].isnull(), 'Episode_Length_minutes'] = predicted_episode_lengths

# # Evaluate the model on the rows where the episode length is available in the test set
# y_test_non_null = y_test.dropna()  # Remove the rows where the target is null
# X_test_non_null = X_test.loc[y_test_non_null.index]  # Get the corresponding features for non-null rows

# # Predict for the non-null rows
# y_pred_non_null = pipeline.predict(X_test_non_null)

# # Evaluate the model on the non-null data
# mae = mean_absolute_error(y_test_non_null, y_pred_non_null)
# mse = mean_squared_error(y_test_non_null, y_pred_non_null)
# r2 = r2_score(y_test_non_null, y_pred_non_null)

# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")

# # The test_cleaned_encoded now has the missing episode lengths predicted and filled



test_cleaned_encoded.info()


train_cleaned_encoded.info()


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------ Data Preparation ------------------

# Categorical columns for one-hot encoding
categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

# Separate features and target
X = train_cleaned_encoded.drop('Listening_Time_minutes', axis=1)
y = train_cleaned_encoded['Listening_Time_minutes']

# Apply same transform to test set (no target)
X_test = test_cleaned_encoded

# ColumnTransformer for one-hot encoding categorical features
column_transformer = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

# Fit on training data
X_encoded = column_transformer.fit_transform(X)

# Convert to dense array if sparse (Keras requires dense input)
if hasattr(X_encoded, "toarray"):
    X_encoded = X_encoded.toarray()

# Transform the test set
X_test_encoded = column_transformer.transform(X_test)
if hasattr(X_test_encoded, "toarray"):
    X_test_encoded = X_test_encoded.toarray()

# Train-test split (80:20)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ------------------ Model Definition ------------------

model = models.Sequential([
    layers.InputLayer(input_shape=(X_train_split.shape[1],)),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),

    layers.Dense(1)  # Output layer for regression
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ------------------ Model Training ------------------

history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=50,
    batch_size=1024,
    callbacks=[early_stop],
    verbose=1
)

# ------------------ Evaluation ------------------

y_pred = model.predict(X_val_split)
rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
print(f"Validation RMSE: {rmse:.2f}")



if hasattr(X_test_encoded, "toarray"):
    X_test_encoded = X_test_encoded.toarray()

test_predictions = model.predict(X_test_encoded)


submission = pd.DataFrame({
    'id': test_cleaned_encoded.index,
    'Listening_Time_minutes': test_predictions.flatten()  # Ensure it's 1D
})
submission.to_csv('/kaggle/working/submission.csv', index=False)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # One-Hot Encoding
# categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

# # Split the features and target for training data
# X_train = train_cleaned_encoded.drop('Listening_Time_minutes', axis=1)
# y_train = train_cleaned_encoded['Listening_Time_minutes']

# # Use the same columns for test (make sure test doesn't include the target!)
# X_test = test_cleaned_encoded  # Should already exclude 'Listening_Time_minutes'

# # Set up the ColumnTransformer
# column_transformer = ColumnTransformer(
#     transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
#     remainder='passthrough'
# )

# # Fit on training data, transform both train and test
# X_train_encoded = column_transformer.fit_transform(X_train)
# X_test_encoded = column_transformer.transform(X_test)


# # Define categorical columns for imputation step
# categorical_cols_pop = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time']

# # Impute Guest_Popularity 
# drop_columns_train = ['Guest_Popularity_percentage', 'Episode_Length_minutes']
# drop_columns_test = ['Guest_Popularity_percentage', 'Episode_Length_minutes']

# # Add Listening_Time_minutes to drop columns if present in train data
# if 'Listening_Time_minutes' in train_cleaned_encoded.columns:
#     drop_columns_train.append('Listening_Time_minutes')

# # Separate rows with non-missing target for training
# train_with_values = train_cleaned_encoded[train_cleaned_encoded['Guest_Popularity_percentage'].notna()]
# X_train_pop = train_with_values.drop(columns=drop_columns_train)
# y_train_pop = train_with_values['Guest_Popularity_percentage']

# # Rows with missing target for prediction (test set)
# test_with_missing_values = test_cleaned_encoded[test_cleaned_encoded['Guest_Popularity_percentage'].isna()]
# X_test_pop = test_cleaned_encoded.drop(columns=drop_columns_test)

# # Create a new transformer for categorical columns
# column_transformer_pop = ColumnTransformer(
#     transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_pop)],
#     remainder='passthrough'  # keep other columns as they are
# )

# # Fit-transform on train, transform on test
# X_train_pop_encoded = column_transformer_pop.fit_transform(X_train_pop)
# X_test_pop_encoded = column_transformer_pop.transform(X_test_pop)

# # Train RandomForestRegressor
# rf_pop = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_pop.fit(X_train_pop_encoded, y_train_pop)

# # Predict missing values
# train_cleaned_encoded.loc[train_cleaned_encoded['Guest_Popularity_percentage'].isna(), 'Guest_Popularity_percentage'] = rf_pop.predict(X_train_pop_encoded)
# test_cleaned_encoded.loc[test_cleaned_encoded['Guest_Popularity_percentage'].isna(), 'Guest_Popularity_percentage'] = rf_pop.predict(X_test_pop_encoded)

# # Check for missing values
# print("Train missing values:\n", train_cleaned_encoded.isnull().sum())
# print("\nTest missing values:\n", test_cleaned_encoded.isnull().sum())


