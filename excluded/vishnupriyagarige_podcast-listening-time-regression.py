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
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
original_data = pd.read_csv(r"/kaggle/input/podcast-data/podcast_dataset.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.head()


train_data.info()


train_data.isna().sum().sort_values(ascending=False)


train_data = train_data.drop_duplicates()
print("shape of the data :",train_data.shape)


# Calculate missing values
missing_values = train_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


original_data.head()


original_data.isna().sum().sort_values(ascending=False)


original_data = original_data.dropna(subset=['Listening_Time_minutes']).drop_duplicates()
original_data.shape


test_data.head()


test_data.isna().sum().sort_values(ascending=False)


# Calculate missing values
missing_values = test_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


train_data = train_data.drop("id", axis=1)
train_data = pd.concat([train_data, original_data],axis=0, ignore_index=True)
print("shape of the data :",train_data.shape)


# Categorical columns to plot
cat_cols = ['Genre', 'Episode_Sentiment', 'Publication_Time', 'Publication_Day']

# Set up 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cat_cols):
    train_data[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()



# Extract episode number
train_data['Episode_Number'] = train_data['Episode_Title'].str.extract(r'(\d+)$').astype(int)
test_data['Episode_Number'] = test_data['Episode_Title'].str.extract(r'(\d+)$').astype(int)

train_data = train_data.drop(['Episode_Title'], axis=1)
test_data = test_data.drop(['Episode_Title'], axis=1)


#train_data = train_data.drop('id', axis = 1)
num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['Listening_Time_minutes']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


len(num_cols), len(num_cols_test), len(cat_cols), len(cat_cols_test)


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object', 'category']).columns] = test_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))


import pandas as pd

def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Removes outliers from all numerical columns using the specified method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        method (str): 'iqr' for Interquartile Range or 'zscore' for Z-score method
        threshold (float): The threshold for defining an outlier (default 1.5 for IQR)
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns  # Select only numeric columns

    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    elif method == 'zscore':
        from scipy.stats import zscore
        df_clean = df_clean[(df_clean[numeric_cols].apply(zscore).abs() < threshold).all(axis=1)]

    return df_clean

# Usage Example
train_data = remove_outliers(train_data, method='iqr')  # Remove outliers from training data
print(train_data.shape)


from sklearn.preprocessing import LabelEncoder

# Select categorical columns automatically
cat_cols = train_data.select_dtypes(include=['object', 'category']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Apply to each categorical column
for col in cat_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])



def create_features(df, genre_avg_length=None,genre_avg_popularity=None, is_train=True):
    # Ads per minute
    df["Ads_per_minute"] = df["Number_of_Ads"] / df["Episode_Length_minutes"]

    # Guest and Host Impact (assumes Episode_Sentiment is numeric)
    df["Guest_Impact"] = df["Guest_Popularity_percentage"] * df["Episode_Sentiment"]
    df["Host_Impact"] = df["Host_Popularity_percentage"] * df["Episode_Sentiment"]

    # Is Weekend
    df["Is_Weekend"] = df["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)

    # Extract episode number
    #df["Episode_Number"] = df["Episode_Title"].str.extract(r'(\d+)$')[0].astype(int)
    df["Popularity_Diff"] = df["Host_Popularity_percentage"] - df["Guest_Popularity_percentage"]
    df["Popularity_Ratio"] = df["Host_Popularity_percentage"] / (df["Guest_Popularity_percentage"] + 1e-5)

    df["Is_Short_Episode"] = (df["Episode_Length_minutes"] < 20).astype(int)
    df["Is_Long_Episode"] = (df["Episode_Length_minutes"] > 60).astype(int)

    # Genre-wise average episode length
    if is_train:
        genre_avg_length = df.groupby("Genre")["Episode_Length_minutes"].mean().to_dict()
    df["Avg_Episode_Length_for_Genre"] = df["Genre"].map(genre_avg_length)

    # Genre-wise average popularity
    if is_train:
        genre_avg_popularity = df.groupby("Genre")[["Host_Popularity_percentage", "Guest_Popularity_percentage"]].mean().mean(axis=1).to_dict()
    df["Avg_Popularity_for_Genre"] = df["Genre"].map(genre_avg_popularity)

    return df, genre_avg_length, genre_avg_popularity


# Apply to both train and test
train_data, genre_avg_length, genre_avg_popularity = create_features(train_data, is_train=True)
test_data, _, _ = create_features(test_data, genre_avg_length=genre_avg_length, genre_avg_popularity=genre_avg_popularity, is_train=False)
train_data.shape, test_data.shape


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols_test] = scaler.transform(test_data[num_cols_test])


# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(15, 3))

# Boxplot
sns.boxplot(x=train_data["Listening_Time_minutes"], ax=axes[0], color='lightblue')
axes[0].set_title("Boxplot of Price")

#  Distribution plot (KDE + Histogram)
sns.histplot(train_data["Listening_Time_minutes"], bins=30, kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title("Histogram of Listening_Time_minutes")

# Show the plots
plt.tight_layout()
plt.show()


X = train_data.drop(['Listening_Time_minutes'], axis=1)
y = train_data['Listening_Time_minutes']
test = test_data.drop(['id'], axis=1)


params = {'n_estimators': 4350, 'max_depth': 14, 'learning_rate': 0.017833840653144653, 'subsample': 0.9345699295236994, 'colsample_bytree': 0.7245363775767917, 'reg_alpha': 0.00014264383052697044, 'reg_lambda': 7.178514151735702e-08, 'gamma': 0.1174177426013003, 'n_jobs':-1}
#value: 12.603070614751548
parameters = {'n_estimators': 4450, 'max_depth': 14, 'learning_rate': 0.010110014169400876, 'subsample': 0.783446781770771, 'colsample_bytree': 0.8909398251993162, 'reg_alpha': 0.5988908047775008, 'reg_lambda': 0.0004557908330224355, 'gamma': 0.0041268709908939445}
#value: 12.603120635509221
parameters1 = {'n_estimators': 2500, 'max_depth': 15, 'learning_rate': 0.01089005065919055, 'subsample': 0.9687835334938029, 'colsample_bytree': 0.6544885310390419, 'reg_alpha': 1.3155809330558803, 'reg_lambda': 3.58194185631273e-07, 'gamma': 1.6258781456503933e-08}
#value: 9.949869808668435.


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
rmse_scores = []
preds = []
# Initialize the model
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **parameters1)
# K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],eval_metric='rmse',
              early_stopping_rounds=100,verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)
    pred = model.predict(test)
    preds.append(pred)
     # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_scores.append(rmse)

# Calculate the average RMSE across all folds
avg_rmse = np.mean(rmse_scores)

# Print RMSE for each fold and the average RMSE
#print("RMSE scores for each fold:", rmse_scores)
print("Average RMSE:", avg_rmse)

xgb_preds = np.mean(preds, axis=0)
submission_xgb = pd.DataFrame({'id': sample_submission.id, 'Listening_Time_minutes': xgb_preds})
print(submission_xgb.head())
submission_xgb.to_csv('submission_xgb.csv', index=False)

submission_xgb['Listening_Time_minutes'].hist()


import xgboost as xgb
from xgboost import plot_importance

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='gain', max_num_features=20)  # Top 20 important features
plt.title("XGBoost Feature Importance (Gain)")
plt.show()

