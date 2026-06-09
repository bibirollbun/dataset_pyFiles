import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Reading .csv data file
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df_train.head()


df_test.head()


df_test.shape,df_train.shape


df_train = df_train.drop(columns=['id'])


df_train.info()


df_test.info()


df_train.isnull().sum()


df_test.isnull().sum()


# Check for duplicate rows
duplicate_rows = df_train.duplicated()
num_duplicate_rows = duplicate_rows.sum()
print(f"Duplicate rows in train set: {num_duplicate_rows}")


# Columns with missing values
missing_num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage','Number_of_Ads']

# Impute missing values using median from training set
for col in missing_num_cols:
    median_val = df_train[col].median()
    df_train[col].fillna(median_val, inplace=True)
    df_test[col].fillna(median_val, inplace=True)


# Drop uninformative high-cardinality features
cols_to_drop = ['Podcast_Name', 'Episode_Title']
df_train.drop(columns=cols_to_drop, inplace=True)
df_test.drop(columns=cols_to_drop, inplace=True)

# Map Publication_Time to 0/1
time_map = {'Morning': 0, 'Evening': 1,'Night':2,'Afternoon':3}
df_train['Publication_Time'] = df_train['Publication_Time'].map(time_map)
df_test['Publication_Time'] = df_test['Publication_Time'].map(time_map)


df_test.shape,df_train.shape


from sklearn.preprocessing import LabelEncoder

# Label Encode Episode_Sentiment
le_sentiment = LabelEncoder()
df_train['Episode_Sentiment'] = le_sentiment.fit_transform(df_train['Episode_Sentiment'])
df_test['Episode_Sentiment'] = le_sentiment.transform(df_test['Episode_Sentiment'])

# Convert boolean columns (from one-hot encoding) to integers (0 for False, 1 for True)
bool_columns_train = [col for col in df_train.columns if df_train[col].dtype == 'bool']
bool_columns_test = [col for col in df_test.columns if df_test[col].dtype == 'bool']

df_train[bool_columns_train] = df_train[bool_columns_train].astype(int)
df_test[bool_columns_test] = df_test[bool_columns_test].astype(int)

# Align columns (to match test and train after encoding)
# df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Drop 'Listening_Time_minutes' column from df_test to avoid having it in the test set
df_test = df_test.drop(columns=['Listening_Time_minutes'], errors='ignore')


df_train.info()


df_test.info()


df_test.shape,df_train.shape


from sklearn.preprocessing import StandardScaler

# Define the numerical columns
numerical_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                     'Guest_Popularity_percentage', 'Number_of_Ads']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the train and test sets
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])

# Check the transformed data (optional)
print(df_train[numerical_columns].head())



df_test.shape,df_train.shape


# Check column data types and summary statistics
print(df_train.dtypes)  # Data types for each column
print(df_train.describe())  # Summary statistics for numerical columns


# Create a binary feature for weekends (1 for Saturday/Sunday, 0 for weekdays)
df_train['Is_Weekend'] = df_train['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
df_test['Is_Weekend'] = df_test['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# One-Hot Encode Genre and Publication_Day
df_train = pd.get_dummies(df_train, columns=['Genre', 'Publication_Day'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Genre', 'Publication_Day'], drop_first=True)

# Check if the columns are transformed correctly
print(df_train[['Publication_Time', 'Is_Weekend']].head())
print(df_test[['Publication_Time', 'Is_Weekend']].head())



df_test.shape,df_train.shape


df_train.info()


# Check target variable distribution
print(df_train['Listening_Time_minutes'].describe())  # Basic statistics for the target


import seaborn as sns
# Correlation with the target variable
correlation_target = df_train.corr()['Listening_Time_minutes'].sort_values(ascending=False)

# Print the correlation values with the target
print(correlation_target)

# Optionally, you can plot the correlations that are important
plt.figure(figsize=(10, 6))
sns.heatmap(df_train[correlation_target.index].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()



# List all numerical columns
numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns

# Check skewness for each numerical column
for col in numerical_columns:
    skewness = df_train[col].skew()
    print(f"Skewness of {col}: {skewness}")



# # Apply log transformation to 'Number_of_Ads' if highly skewed
# df_train['log_number_of_ads'] = np.log1p(df_train['Number_of_Ads'])
# df_test['log_number_of_ads'] = np.log1p(df_test['Number_of_Ads'])


# Add the interaction feature for training data
df_train['episode_popularity_interaction'] = df_train['Episode_Length_minutes'] * df_train['Host_Popularity_percentage']

# Add the same interaction feature for test data
df_test['episode_popularity_interaction'] = df_test['Episode_Length_minutes'] * df_test['Host_Popularity_percentage']



# Check for missing values in the dataset
print(df_train.isnull().sum())

# Check for skewness in numerical columns (values greater than 1 or less than -1 may need transformation)
print(df_train.skew())  



df_test.shape,df_train.shape


import pandas as pd

# Assuming your features are in 'df_train' and your target variable is 'Listening_Time_minutes'
X = df_train.drop(columns=['Listening_Time_minutes'])  # Drop the target column
y = df_train['Listening_Time_minutes']


from sklearn.model_selection import KFold

# Define K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# Initialize the model
model = XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200)

# Lists to store MSE scores
mse_scores = []



# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(df_train):
    X_train, X_val = df_train.drop(columns=['Listening_Time_minutes']).iloc[train_index], df_train.drop(columns=['Listening_Time_minutes']).iloc[val_index]
    y_train, y_val = df_train['Listening_Time_minutes'].iloc[train_index], df_train['Listening_Time_minutes'].iloc[val_index]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on validation set
    val_predictions = model.predict(X_val)

    # Calculate MSE for the fold
    mse = mean_squared_error(y_val, val_predictions)
    mse_scores.append(mse)

# Display MSE scores for each fold
print("MSE Scores for Each Fold:", mse_scores)


# Calculate Mean MSE
mean_mse = np.mean(mse_scores)
print("Mean MSE:", mean_mse)


# Calculate RMSE
mean_rmse = np.sqrt(mean_mse)
print("Mean RMSE:", mean_rmse)


df_test.info()


# from sklearn.model_selection import GridSearchCV

# # Define the hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5],
#     'learning_rate': [0.01, 0.1]
# }

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf)

# # Fit the grid search
# grid_search.fit(df_train.drop(columns=['Listening_Time_minutes']), df_train['Listening_Time_minutes'])

# # Get the best hyperparameters
# print("Best Hyperparameters:", grid_search.best_params_)



# Train the model on the full training data
final_model = XGBRegressor()  # Make sure to use the model you chose
final_model.fit(df_train.drop(columns=['Listening_Time_minutes']), df_train['Listening_Time_minutes'])

# Predict on the test set (drop the target column if it's in the test set)
df_test['Listening_Time_minutes'] = final_model.predict(df_test.drop(columns=['id']))

# Create the submission file
submission = df_test[['id', 'Listening_Time_minutes']]  # Assuming 'id' is the identifier column in the test set
submission.to_csv('submission.csv', index=False)

print("Submission file has been created.")





