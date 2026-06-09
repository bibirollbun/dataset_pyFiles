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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_train.head()


df_test.head()


# !pip install --upgrade pip
# !pip install --upgrade skimpy




# from skimpy import skim

# skim(df_train)


df_train.drop(columns=['id'],inplace=True)
df_train['Number_of_Ads'] = df_train['Number_of_Ads'].fillna(df_train['Number_of_Ads'].mode()[0])
# Find the maximum and 5 lowest values
max_value = df_train['Episode_Length_minutes'].max()
lowest_values = df_train['Episode_Length_minutes'].nsmallest(5).tolist()

# Filter out rows with these values
df_train = df_train[~df_train['Episode_Length_minutes'].isin([max_value] + lowest_values)]


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df_train['Listening_Time_minutes'], bins=30, kde=True, color='blue')
plt.title("Distribution of Listening Time (minutes)")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.show()


categorical_cols = df_train.select_dtypes(include=['object']).columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))
fig.suptitle("Categorical Feature Distributions", fontsize=16)
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    if i < len(axes):  
        sns.countplot(data=df_train, x=col, order=df_train[col].value_counts().index, palette="viridis", ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].tick_params(axis='x', rotation=90)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))
fig.suptitle("Listening Time vs Categorical Features", fontsize=12)
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    if i < len(axes): 
        sns.boxplot(data=df_train, x=col, y='Listening_Time_minutes', palette="Set2", ax=axes[i])
        axes[i].set_title(f"{col} vs Listening Time")
        axes[i].tick_params(axis='x', rotation=90)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %load_ext cudf.pandas 
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler
# import cudf
# import pandas as pd
# from cuml.preprocessing import StandardScaler as cumlStandardScaler

# numerical_features = [
#     "Episode_Length_minutes",
#     "Host_Popularity_percentage",
#     "Guest_Popularity_percentage",
#     "Number_of_Ads",
#     "Listening_Time_minutes",
# ]

# # 1. Convert pandas DataFrame to cuDF DataFrame
# df3_gpu = cudf.from_pandas(df_train)

# # 2. Define batch size
# batch_size = 10000  # Adjust this based on available memory

# # 3. Process in batches
# for i in range(0, len(df3_gpu), batch_size):
#     batch = df3_gpu[i : i + batch_size]

#     # Apply StandardScaler (using cuML StandardScaler)
#     scaler = cumlStandardScaler()
#     scaled_batch = scaler.fit_transform(batch[numerical_features])

#     # Apply KNN Imputer (using CPU version)
#     imputer = KNNImputer(n_neighbors=4)
#     imputed_scaled_batch = imputer.fit_transform(scaled_batch.to_pandas())

#     # Convert imputed_scaled_batch back to cuDF DataFrame with correct column names
#     imputed_scaled_batch_df = cudf.DataFrame(imputed_scaled_batch, columns=numerical_features) 

#     # Inverse transform the scaled data (using cuML StandardScaler)
#     imputed_batch = scaler.inverse_transform(imputed_scaled_batch_df)
    
#     # Assign column names to imputed_batch
#     imputed_batch.columns = numerical_features  

#     # Update missing values in 'Episode_Length_minutes' for the batch
#     # Use .iloc to select rows by position for the batch only
#     df3_gpu.iloc[batch.index, df3_gpu.columns.get_loc('Episode_Length_minutes')] = imputed_batch['Episode_Length_minutes']

# # 4. Convert cuDF DataFrame back to pandas DataFrame
# df_train = df3_gpu.to_pandas()


df_train_new = pd.read_csv('/kaggle/input/knn-predicted/df3.csv')


df_train_new


df_train_new['Guest_Popularity_percentage'].fillna(df_train_new['Guest_Popularity_percentage'].median(), inplace=True)



df_train_new.isnull().sum(),df_test.isnull().sum()


df_train_new['Number_of_Ads'] = df_train_new['Number_of_Ads'].fillna(df_train_new['Number_of_Ads'].mode()[0])


from sklearn.preprocessing import LabelEncoder

categorical_cols = df_train_new.select_dtypes(include=['object']).columns
label_encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df_train_new[col] = le.fit_transform(df_train_new[col])
    
    label_encoders[col] = le  
df_train_new = df_train_new.astype(float)


print("Categorical columns converted to numerical successfully!")


def feature_engineering(df, is_train=True):
    # Convert Publication_Time to numeric
    df['Publication_Time'] = pd.to_numeric(df['Publication_Time'], errors='coerce')

    df['Is_Weekend'] = df['Publication_Day'].apply(lambda x: 1 if x in [6, 7] else 0)

    df['Daypart'] = df['Publication_Time'].apply(lambda x: 
        'Morning' if 6 <= x < 12 else 
        'Afternoon' if 12 <= x < 18 else 
        'Evening' if 18 <= x < 24 else 'Night'
    )
    df['Daypart'] = df['Daypart'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3})

    df['Host_Guest_Popularity_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-5)
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-5)
    df['Popularity_Score'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2  
    df['Long_Episode'] = (df['Episode_Length_minutes'] > 75).astype(int)
    df['Highly_Popular_Host'] = (df['Host_Popularity_percentage'] > 75).astype(int)
    df['Highly_Popular_Guest'] = (df['Guest_Popularity_percentage'] > 75).astype(int)
    df['Host_Guest_Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Host_Guest_Popularity_Sum'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Ad_Impact'] = df['Number_of_Ads'] * df['Episode_Length_minutes']

    df['Episode_Length_Bin'] = pd.cut(df['Episode_Length_minutes'],
                                      bins=[-1, 187500, 375000, 562500, np.inf],
                                      labels=[0, 1, 2, 3])  
    df['Episode_Length_Bin'] = df['Episode_Length_Bin'].astype(int)

    df['High_Ad_Load'] = (df['Number_of_Ads'] > 2).astype(int)

    return df


df_train_new = feature_engineering(df_train_new, is_train=True)


print("✅ Feature Engineering Complete!")


y = df_train_new['Listening_Time_minutes'] 
df_train_new = df_train_new.drop(['Listening_Time_minutes'],axis=1)


df_train_new


common_cols = df_train.columns.intersection(df_test.columns)

numerical_cols = df_train[common_cols].select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_train[common_cols].select_dtypes(include=['object']).columns


df_test[numerical_cols] = df_test[numerical_cols].fillna(df_train[numerical_cols].median())

df_test[categorical_cols] = df_test[categorical_cols].apply(lambda x: x.fillna(df_train[x.name].mode()[0]))


from sklearn.preprocessing import LabelEncoder

categorical_cols = df_train.select_dtypes(include=['object']).columns
label_encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col]) 
    label_encoders[col] = le  

df_test = df_test.astype(float)

print("Categorical columns converted to numerical successfully!")


df_test = feature_engineering(df_test, is_train=False)


import xgboost as xgb


from sklearn.model_selection import KFold


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

xgb_params = {
    'n_estimators': 500,
    'max_depth': 14,
    'learning_rate': 0.0395,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist', 
    'n_jobs': -1  
}


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(len(df_test)) 

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_new, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")    
    X_train, X_val = df_train_new.iloc[train_idx], df_train_new.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]   
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(df_train_new, y, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)    
    val_pred = model.predict(X_val)
    score = rmse(y_val, val_pred)
    scores.append(score)
    test_preds += model.predict(df_test) / n_splits      
    print(f"Fold {fold + 1} RMSE: {score:.4f}")
print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Feature Importance (XGBoost)")
axes[0, 0].set_xlabel("Importance Score")
axes[0, 0].set_ylabel("Features")

sns.scatterplot(x=y_val, y=val_pred, alpha=0.6, edgecolors="k", ax=axes[0, 1])
axes[0, 1].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '--r', linewidth=2) 
axes[0, 1].set_title("Actual vs. Predicted Listening Time")
axes[0, 1].set_xlabel("Actual Values")
axes[0, 1].set_ylabel("Predicted Values")

residuals = y_val - val_pred
sns.histplot(residuals, bins=30, kde=True, color='blue', ax=axes[1, 0])
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Frequency")

sns.histplot(test_preds, bins=30, kde=True, color='green', ax=axes[1, 1])
axes[1, 1].set_title("Test Predictions Distribution")
axes[1, 1].set_xlabel("Predicted Listening Time")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


df_sub.head()


df_sub['Listening_Time_minutes'] = test_preds


df_sub.to_csv('submission.csv', index=False)


df_sub['Listening_Time_minutes'].hist()




