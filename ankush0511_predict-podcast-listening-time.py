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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
import category_encoders as ce
from sklearn.ensemble import VotingRegressor


df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df.head()


test.head()


df.shape


df.columns


df.isnull().sum()


test.isnull().sum()


df.duplicated().sum()


test.duplicated().sum()


df.info()


test.info()


df.describe()


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Episode_Length_minutes', palette='viridis',
            linewidth=2.5,width=0.4)

plt.title('Distribution of Episode Length', fontsize=16)
plt.xlabel('Episode Length (minutes)', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=test, x='Episode_Length_minutes', palette='viridis',
            linewidth=2.5,width=0.4)

plt.title('Distribution of Episode Length', fontsize=16)
plt.xlabel('Episode Length (minutes)', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Host_Popularity_percentage', palette='viridis',
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Host Popularity', fontsize=16)
plt.xlabel('Host Popularity Percentage', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=test, x='Host_Popularity_percentage', palette='viridis',
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Host Popularity', fontsize=16)
plt.xlabel('Host Popularity Percentage', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Guest_Popularity_percentage', palette='viridis',
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Guest Popularity', fontsize=16)
plt.xlabel('Guest Popularity Percentage', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Number_of_Ads', palette='viridis',
            # showfliers=False,  # Remove outliers
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Number_of_Ads', fontsize=16)
plt.xlabel('Number_of_Ads', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=test, x='Number_of_Ads', palette='viridis',
            # showfliers=False,  # Remove outliers
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Number_of_Ads TEST', fontsize=16)
plt.xlabel('Number_of_Ads', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Listening_Time_minutes', palette='viridis',
            linewidth=2.5,   # Thicker box lines
            width=0.4)      # Adjust box width

plt.title('Distribution of Listening_Time_minutes', fontsize=16)
plt.xlabel('Listening Time minutes', fontsize=12)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(12, 9))
sns.boxplot(data=df, x='Listening_Time_minutes', y='Publication_Day', hue='Publication_Time',
            palette="Set3", linewidth=2.5, width=0.8, fliersize=4)
plt.title('Listening Time by Publication Day and Time', fontsize=16)
plt.xlabel('Listening Time (minutes)', fontsize=12)
plt.ylabel('Publication Day', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Publication Time', loc='upper right',bbox_to_anchor=(1.23, 1))
plt.xticks(rotation=45, ha='right')
plt.show()


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Genre', palette='viridis')

# Add annotations and styling
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Genres', fontsize=16)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 8))

ax = sns.countplot(data=df, x='Podcast_Name', palette='rocket')

# Add annotations and styling
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Podcast_Name', fontsize=16)
plt.xlabel('Podcast_Name', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Publication_Day', palette='viridis')

# Add annotations and styling
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Publication_Day', fontsize=16)
plt.xlabel('Publication_Day', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Publication_Time', palette='viridis')

# Add annotations and styling
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Publication_Time', fontsize=16)
plt.xlabel('Publication_Time', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Episode_Sentiment', palette='viridis')

# Add annotations and styling
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Episode Sentiment', fontsize=16)
plt.xlabel('Episode Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            linecolor='black',
            cbar_kws={'shrink': 0.8}  # Adjust colorbar size
           )

plt.title('Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


sns.histplot(data=df,x='Episode_Length_minutes',kde=True)


Q1 = df['Listening_Time_minutes'].quantile(0.25)
Q3 = df['Listening_Time_minutes'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound only
upper_bound = Q3 + 1.5 * IQR


outlier = df["Episode_Length_minutes"] > upper_bound
df.loc[outlier, "Episode_Length_minutes"] = df["Episode_Length_minutes"].mean() #outlier replace by mean


Q1 = df['Number_of_Ads'].quantile(0.25)
Q3 = df['Number_of_Ads'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound only
upper_bound = Q3 + 1.5 * IQR
outlier = df["Number_of_Ads"] > upper_bound
df.loc[outlier, "Number_of_Ads"] = df["Number_of_Ads"].mean() #outlier replace by mean


Q1 = test['Number_of_Ads'].quantile(0.25)
Q3 = test['Number_of_Ads'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound only
upper_bound = Q3 + 1.5 * IQR
outlier = test["Number_of_Ads"] > upper_bound
test.loc[outlier, "Number_of_Ads"] = test["Number_of_Ads"].mean() #outlier replace by mean


Q1 = test['Episode_Length_minutes'].quantile(0.25)
Q3 = test['Episode_Length_minutes'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound only
upper_bound = Q3 + 1.5 * IQR
outlier = test["Episode_Length_minutes"] > upper_bound
test.loc[outlier, "Episode_Length_minutes"] = test["Episode_Length_minutes"].mean() #outlier replace by mean


df.isnull().sum()


df.isnull().sum()/df.shape[0]*100


df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(),inplace=True)
df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(),inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(),inplace=True)
print(df.isnull().sum())


test.isnull().sum()


test.isnull().sum()/test.shape[0]*100


test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(),inplace=True)
test['Number_of_Ads'].fillna(test['Number_of_Ads'].median(),inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(),inplace=True)
print(test.isnull().sum())


df['total_popularity']=df['Guest_Popularity_percentage']+df['Host_Popularity_percentage']
df['popularity_difference']=df['Guest_Popularity_percentage']-df['Host_Popularity_percentage']


test['total_popularity']=test['Guest_Popularity_percentage']+test['Host_Popularity_percentage']
test['popularity_difference']=test['Guest_Popularity_percentage']-test['Host_Popularity_percentage']


test.columns


df.columns


x=df.drop(['id','Podcast_Name','Episode_Title','Listening_Time_minutes'],axis=1)
y=df['Listening_Time_minutes']
x_test=test.drop(['id','Podcast_Name','Episode_Title'],axis=1)


num_col=x.select_dtypes(include=["float64", "int64"]).columns.tolist()

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x[num_col])
x_test_scaled=scaler.transform(x_test[num_col])


x_scaled_df=pd.DataFrame(x_scaled,columns=num_col)
x_test_scaled_df=pd.DataFrame(x_test_scaled,columns=num_col)


categorical_cols=x.select_dtypes(include=["object"]).columns.tolist()
pref=['gen','day','time','sent']
x_train_enc=pd.get_dummies(data = x ,columns =  categorical_cols, prefix = pref,drop_first = True)
x_test_enc=pd.get_dummies(data = x_test ,columns =  categorical_cols, prefix = pref,drop_first = True)


bool_columns = x_train_enc.select_dtypes(include = "bool").columns
x_train_enc[bool_columns] = x_train_enc[bool_columns].astype(int)

bool_columns = x_test_enc.select_dtypes(include = "bool").columns
x_test_enc[bool_columns] = x_test_enc[bool_columns].astype(int)


x_train_enc


X_categorical = x_train_enc.drop(num_col, axis=1)
X_train_final = pd.concat([x_scaled_df, X_categorical.reset_index(drop=True)], axis=1)

X_categorical_test = x_test_enc.drop(num_col, axis=1)
X_scaled_test_final = pd.concat([x_test_scaled_df, X_categorical_test.reset_index(drop=True)], axis=1)


lr=LinearRegression()
lr.fit(X_train_final,y)
# y_pred=lr.predict(X_scaled_test_final)


lr_pred=lr.predict(X_scaled_test_final)


rmse = np.sqrt(mean_squared_error(y,lr.predict(X_train_final)))


rmse


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": lr_pred
})

submission.to_csv("submission_LinearR.csv", index=False)


ridge=Ridge(alpha=0.08)
ridge.fit(X_train_final,y)
pred=ridge.predict(X_scaled_test_final)
rmse=np.sqrt(mean_squared_error(y,ridge.predict(X_train_final)))
print(rmse)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": lr_pred
})

submission.to_csv("submission_Ridge.csv", index=False)


lass0=Lasso(alpha=0.01,random_state=42,max_iter=1000)
lass0.fit(X_train_final,y)
pred=lass0.predict(X_scaled_test_final)
rmse=np.sqrt(mean_squared_error(y,lass0.predict(X_train_final)))
print(rmse)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": lr_pred
})

submission.to_csv("submission_Lasso.csv", index=False)


 model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


model.fit(X_train_final, y)


pred=model.predict(X_scaled_test_final)
rmse=np.sqrt(mean_squared_error(y,model.predict(X_train_final)))
print(rmse)


xgb_params = {
    'n_estimators': 400,
    'max_depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}


xgb=XGBRegressor(**xgb_params)
xgb.fit(X_train_final,y)


pred=xgb.predict(X_scaled_test_final)
rmse=np.sqrt(mean_squared_error(y,xgb.predict(X_train_final)))
print(rmse)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": lr_pred
})

submission.to_csv("submission_Xgb.csv", index=False)


xgb_model = XGBRegressor(random_state = 42,colsample_bytree = 0.7,gamma =1,learning_rate = 0.05,max_depth= 10,n_estimators = 1000,subsample = 0.8,reg_lambda = 0.1,reg_alpha = 10)
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)


voting = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('mlp', ann_model)
    ],
    n_jobs=-1
)

voting.fit(X_train_final, y)
y_test_pred = voting.predict(X_scaled_test_final)


rmse=np.sqrt(mean_squared_error(y,voting.predict(X_train_final)))
print(rmse)




