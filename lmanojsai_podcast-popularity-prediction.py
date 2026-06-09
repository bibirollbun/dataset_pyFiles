# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Reading the dataframe
df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df.head()


df.info()


# Checking for missing values
df.isnull().sum()


# Statistical summary
df.describe()


podcast_name_freq = df["Podcast_Name"].value_counts()
top_10_podcasts = podcast_name_freq.nlargest(10).index.to_list()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Podcast_Name", order=top_10_podcasts)
plt.title("Top 10 most frequent podcasts")
plt.xlabel("Podcast Name")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


top_10_genre = df["Genre"].value_counts().nlargest(10).index.to_list()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Genre", order=top_10_genre)
plt.xlabel("Genre")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Publication_Day")
plt.xlabel("Publication Day")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data=df, x="Episode_Length_minutes", kde=True, ax=axes[0])
axes[0].set_title('Histogram of Episode Length')
axes[0].set_xlabel('Episode Length (minutes)')

sns.boxplot(data=df, y="Episode_Length_minutes", ax=axes[1])
axes[1].set_title('Boxplot of Episode Length')
axes[1].set_ylabel('Episode Length (minutes)')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data=df, x="Listening_Time_minutes", kde=True, ax=axes[0])
axes[0].set_title('Histogram of Listening Time')
axes[0].set_xlabel('Listening Time (minutes)')

sns.boxplot(data=df, y="Listening_Time_minutes", ax=axes[1])
axes[1].set_title('Boxplot of Listening Time')
axes[1].set_ylabel('Listening Time (minutes)')

plt.tight_layout()
plt.show()


sns.scatterplot(data=df, x="Episode_Length_minutes", y="Listening_Time_minutes")


plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Genre", y="Listening_Time_minutes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Publication_Day", y="Listening_Time_minutes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


sns.heatmap(pd.crosstab(df["Genre"],df["Episode_Sentiment"]))


df['episode_number_str'] = df['Episode_Title'].str.extract(r'(\d+)', expand=False)
df['episode_number'] = pd.to_numeric(df['episode_number_str'], errors='coerce')

df.drop(columns=['episode_number_str', 'Episode_Title', 'Podcast_Name'], inplace=True)


df.dropna(subset=['Number_of_Ads'], inplace=True)

q1 = df["Number_of_Ads"].quantile(0.25)
q3 = df["Number_of_Ads"].quantile(0.75)
iqr = q3 - q1
minimum = q1 - 1.5 * iqr
maximum = q3 + 1.5 * iqr
outliers_condition = (df["Number_of_Ads"] < minimum) | (df["Number_of_Ads"] > maximum)
df_no_outliers = df[~outliers_condition]


X = df_no_outliers.drop(columns = ["Listening_Time_minutes", "id"])
y = df_no_outliers["Listening_Time_minutes"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

categorical_cols = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
numeric_cols_with_missing = ["Episode_Length_minutes", "Guest_Popularity_percentage"]
numeric_cols_complete = ["Host_Popularity_percentage", "Number_of_Ads", "episode_number"]

transformer = ColumnTransformer([
   
    ('ohe', OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore'), categorical_cols),
    ('num_impute_scale', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols_with_missing),
    ('num_scale', StandardScaler(), numeric_cols_complete)
], 
    remainder='drop',
    n_jobs=4,
    verbose_feature_names_out=False  
)


pipe = Pipeline([
    ('ct', transformer),
    ('model', XGBRegressor(
        tree_method='hist',     
        random_state=42,
        n_jobs=4,
        gpu_hist=False,        
        predictor='cpu_predictor'
    )),
])


param_distributions = {
    'model__max_depth': randint(3, 8),           
    'model__learning_rate': uniform(0.01, 0.1),  
    'model__n_estimators': randint(100, 300),    
    'model__subsample': uniform(0.7, 0.3),       
    'model__colsample_bytree': uniform(0.7, 0.3), 
    'model__min_child_weight': randint(1, 5),     
    'model__gamma': uniform(0, 0.5),              
    'model__reg_alpha': uniform(0, 0.5),          
    'model__reg_lambda': uniform(1, 2),           
}

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=10,                    
    cv=3,                         
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1,                    
    pre_dispatch='2*n_jobs',      
    error_score='raise',          
    return_train_score=False      
)

print("Starting Hyperparameter Tuning with RandomizedSearchCV...")
random_search.fit(X_train, y_train)
print("Hyperparameter Tuning Finished.")

print("\nBest Parameters Found:")
print(random_search.best_params_)
print(f"\nBest Cross-Validation Score (Negative MSE): {random_search.best_score_:.4f}")

best_pipe = random_search.best_estimator_

print("\nEvaluating the Best Model on the Test Set...")
y_pred_test = best_pipe.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nTest Set Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Set Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Test Set RÂ² Score: {r2:.4f}")


test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_id = test["id"].copy()

test['episode_number_str'] = test['Episode_Title'].str.extract(r'(\d+)', expand=False)
test['episode_number'] = pd.to_numeric(test['episode_number_str'], errors='coerce')

test.drop(columns=["id", "Podcast_Name", "Episode_Title", "episode_number_str"], axis=1, inplace=True)

test_preds = best_pipe.predict(test)
submission_df = pd.DataFrame({
    "id": test_id,
    "Listening_Time_minutes": test_preds
})

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())

