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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.shape,test_df.shape


columns_train = train_df.columns
columns_train


train_df.describe()


numerical_columns = ['id','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage',
                    'Number_of_Ads','Listening_Time_minutes']


for cols in columns_train:
    if cols not in numerical_columns:
        print(f"{cols}-> {train_df[cols].unique()}")


duplicates_row_train = train_df[train_df.duplicated()]
len(duplicates_row_train)


duplicates_row_train = test_df[test_df.duplicated()]
len(duplicates_row_train)


train_df.isnull().sum()


train_df.sample(5)


# Step 1: Calculate group-wise median
group_median_df = train_df.groupby(['Podcast_Name', 'Genre'])['Episode_Length_minutes'].median().reset_index()

# Step 2: Rename the median column
group_median_df.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

# Step 3: Merge the group medians back to the original dataframe
train_df = train_df.merge(group_median_df, on=['Podcast_Name', 'Genre'], how='left')

train_df.head()
# Step 4: Fill missing values using the group median
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Group_Median'])

# Step 5: Drop the helper column
train_df.drop(columns=['Group_Median'], inplace=True)


group_median_df.head()


train_df.isnull().sum()


median_value_left = train_df.groupby('Podcast_Name')['Episode_Length_minutes'].median()


median_value_left


group_median = train_df.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

train_df = train_df.merge(group_median, on='Podcast_Name', how='left')
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Group_Median'])
train_df.drop(columns=['Group_Median'], inplace=True)



train_df.isnull().sum()


test_df.isnull().sum()


group_median_test = test_df.groupby(['Podcast_Name','Genre'])['Episode_Length_minutes'].median().reset_index()
group_median_test.rename(columns = {'Episode_Length_minutes': 'Group_median'},inplace = True)
test_df = test_df.merge(group_median_test,on = ['Podcast_Name','Genre'],how = 'left')
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Group_median'])
test_df.drop(columns = ['Group_median'],inplace = True)


group_median_test.head()


test_df.isnull().sum()


group_median = test_df.groupby('Podcast_Name')['Episode_Length_minutes'].median().reset_index()
group_median.rename(columns={'Episode_Length_minutes': 'Group_Median'}, inplace=True)

test_df = test_df.merge(group_median, on='Podcast_Name', how='left')
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Group_Median'])
test_df.drop(columns=['Group_Median'], inplace=True)



test_df.isnull().sum()


train_df.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Get correlation matrix for numeric columns
corr_matrix = train_df.corr(numeric_only=True)

# Display heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()



group_median = train_df.groupby('Podcast_Name')['Guest_Popularity_percentage'].median().reset_index()
group_median.rename(columns={'Guest_Popularity_percentage': 'Group_Median'}, inplace=True)

train_df = train_df.merge(group_median, on='Podcast_Name', how='left')
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Group_Median'])
train_df.drop(columns=['Group_Median'], inplace=True)



train_df.isnull().sum()


cols = test_df.columns
cols


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Number_of_Ads', y='Listening_Time_minutes', data=train_df)
plt.xticks(rotation=45)
plt.title("Number_of_Ads by Genre")
plt.show()
    


train_df['Number_of_Ads'] = train_df['Number_of_Ads'].apply(lambda x:min(x,3))


train_df[train_df['Number_of_Ads'] > 3]


categorical_cols = []
numerical_columns = ['id','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage',
                    'Number_of_Ads']
for col in cols:
    if col not in numerical_columns:
        categorical_cols.append(col)

print(categorical_cols)


for col in categorical_cols:
    print(f"{col} -> {train_df[col].unique()}")


genre_to_category = {
    'True Crime': 'Entertainment',
    'Comedy': 'Entertainment',
    'Music': 'Entertainment',
    'Sports': 'Entertainment',
    'Lifestyle': 'Entertainment',  # or 'Wellness'

    'Education': 'Knowledge',
    'Technology': 'Knowledge',
    'Business': 'Knowledge',

    'Health': 'Wellness',

    'News': 'Current Affairs'
}



train_df['category'] = train_df['Genre'].map(genre_to_category)
train_df.head()


test_df['category'] = test_df['Genre'].map(genre_to_category)
test_df.head()


train_df = train_df.drop(columns = ['Genre'])


train_df.head()


test_df = test_df.drop(columns = ['Genre'])


day_to_type = {
    'Monday': 'Weekday',
    'Tuesday': 'Weekday',
    'Wednesday': 'Weekday',
    'Thursday': 'Weekday',
    'Friday': 'Weekday',
    'Saturday': 'Weekend',
    'Sunday': 'Weekend'
}
train_df['day_type'] = train_df['Publication_Day'].map(day_to_type)


test_df['day_type'] = test_df['Publication_Day'].map(day_to_type)


train_df.head()


train_df = train_df.drop(columns = ['Publication_Day'])
test_df = test_df.drop(columns = ['Publication_Day'])


train_df.head()


train_df = train_df.drop(columns = ['Podcast_Name','Episode_Title'])


test_df = test_df.drop(columns = ['Podcast_Name','Episode_Title'])


train_df.head()


train_df['Publication_Time'].unique()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['Publication_Time_encoded'] = le.fit_transform(train_df['Publication_Time'])



test_df['Publication_Time_encoded'] = le.transform(test_df['Publication_Time'])


train_df = train_df.drop(columns = ['Publication_Time'])
test_df = test_df.drop(columns = ['Publication_Time'])


train_df.head()


cols = train_df.columns
categorical_cols = []
numerical_columns = ['id','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage',
                    'Number_of_Ads']
for col in cols:
    if col not in numerical_columns:
        categorical_cols.append(col)
        print(f"{col}-->> {train_df[col].unique()}")

print(categorical_cols)


category_map = {
    'Entertainment': 0,
    'Knowledge': 1,
    'Wellness': 2,
    'Current Affairs': 3
}

day_type_map = {
    'Weekday': 0,
    'Weekend': 1
}

sentiment_map = {
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2
}

train_df['category_encoded'] = train_df['category'].map(category_map)
train_df['day_type_encoded'] = train_df['day_type'].map(day_type_map)
train_df['sentiment_encoded'] = train_df['Episode_Sentiment'].map(sentiment_map)



test_df['category_encoded'] = test_df['category'].map(category_map)
test_df['day_type_encoded'] = test_df['day_type'].map(day_type_map)
test_df['sentiment_encoded'] = test_df['Episode_Sentiment'].map(sentiment_map)


train_df.head()


train_df = train_df.drop(columns = ['category','day_type','Episode_Sentiment'])
test_df = test_df.drop(columns = ['category','day_type','Episode_Sentiment'])


train_df.shape,test_df.shape


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


X = train_df.drop(columns = ['id','Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']
cols_to_scale = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[cols_to_scale])
X[cols_to_scale] = X_scaled
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
X_train.shape,X_test.shape


test_new_df =test_df.drop(columns = 'id')
test_scaled = scaler.transform(test_new_df[cols_to_scale])
test_new_df[cols_to_scale] = test_scaled


X_train.head()


X_train.isnull().sum()


X_train['Number_of_Ads'] = X_train['Number_of_Ads'].fillna(3)


X_train.head()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


model  = LinearRegression()


model.fit(X_train,y_train)


y_predict = model.predict(X_test)
print(mean_squared_error(y_test,y_predict))
print(r2_score(y_test,y_predict))


# import optuna
# import xgboost as xgb
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score

# def objective(trial):
#     # Define the hyperparameters to optimize
#     param = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
#         'random_state': 42
#     }
    
#     # For regression tasks
#     model = xgb.XGBRegressor(**param)
    
#     # Perform cross-validation with negative mean squared error
#     # (negative because optuna maximizes the objective by default)
#     scores = cross_val_score(
#         model, X_train, y_train, 
#         cv=5, 
#         scoring='neg_mean_squared_error'
#     )
    
#     # Return the mean negative MSE (to be maximized)
#     return scores.mean()

# # Create a study object and optimize
# study = optuna.create_study(direction='maximize')  # Maximize negative MSE
# study.optimize(objective, n_trials=5)  # You can adjust the number of trials

# # Print results
# print('Number of finished trials:', len(study.trials))
# print('Best trial:')
# trial = study.best_trial
# print('  Value (neg_mean_squared_error):', trial.value)
# print('  Root mean squared error:', np.sqrt(-trial.value))  # Convert back to RMSE
# print('  Params:')
# for key, value in trial.params.items():
#     print(f'    {key}: {value}')

# # Train the model with the best parameters
# best_params = study.best_params
# best_model = xgb.XGBRegressor(**best_params)
# best_model.fit(X_train, y_train)

# # Evaluate on test set
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Test Results with best parameters:")
# print(f"  MSE: {mse:.4f}")
# print(f"  RMSE: {rmse:.4f}")
# print(f"  R²: {r2:.4f}")

# # Feature importance
# if hasattr(X_train, 'columns'):  # If X_train is a DataFrame
#     feature_names = X_train.columns
#     importance = best_model.feature_importances_
    
#     # Sort feature importances
#     indices = np.argsort(importance)[::-1]
    
#     print("\nFeature Importance:")
#     for i, idx in enumerate(indices[:10]):  # Print top 10 features
#         print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

# # You can also visualize the results using matplotlib
# import matplotlib.pyplot as plt

# # Plot feature importance
# if hasattr(X_train, 'columns'):
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(min(10, len(importance))), 
#             [importance[i] for i in indices[:10]],
#             align='center')
#     plt.xticks(range(min(10, len(importance))), 
#                [feature_names[i] for i in indices[:10]], 
#                rotation=90)
#     plt.title('Top 10 Feature Importance')
#     plt.tight_layout()
#     plt.show()


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define best parameters from Optuna
best_params = {
    'n_estimators': 199,
    'max_depth': 10,
    'learning_rate': 0.19103338750042298,
    'subsample': 0.8748436417738474,
    'colsample_bytree': 0.5517907487739219,
    'min_child_weight': 2,
    'gamma': 3.90881702483832,
    'reg_alpha': 1.6714663752270482,
    'reg_lambda': 3.5931017049044884,
    'random_state': 42
}

# Train XGBoost Regressor
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Evaluate (Optional - only if y_test is available)
if 'y_test' in locals():
    y_pred_eval = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_eval)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_eval)
    print(f"Evaluation on Test Set:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")


y_pred_test = model.predict(test_new_df)

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'id': test_df['id'],  # Assuming you have an 'id' column in test_data
    'Listening_Time_minutes': y_pred_test  # 'target' is the name of the prediction column
})

# Save the submission file as a CSV
submission.to_csv('submissionL.csv', index=False)
print("Submission file saved as 'submissionLa.csv'.")




