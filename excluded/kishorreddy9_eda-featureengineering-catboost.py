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


import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.impute import SimpleImputer

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

print("***************** Train ********************")
print(f"Number of rows : {train.shape[0]}")
print(f"Number of columns : {train.shape[1]}")

print("***************** Test *********************")
print(f"Number of rows : {test.shape[0]}")
print(f"Number of columns : {test.shape[1]}")
print("*********************************************")


train.head()


print("******************************* Train ************************************")
print(train.info())
print("******************************* Test ************************************")
print(test.info())


# Null Values
train_null_count = train.isna().sum()
train_null_percent = train_null_count / train.shape[0] * 100
train_dtypes = train.dtypes

test_null_count = test.isna().sum()
test_null_percent = test_null_count / test.shape[0] * 100
test_dtypes = test.dtypes

summary_df = pd.DataFrame({
    "Train Null Count": train_null_count,
    "Train Null Percent": train_null_percent,
    "Train Data Type": train_dtypes,
    "Test Null Count": test_null_count,
    "Test Null Percent": test_null_percent,
    "Test Data Type": test_dtypes
})
summary_df


# Check for duplicate rows
train.duplicated().sum()


# Number of unique values in columns
train.nunique()


train.describe().style.background_gradient("summer")


test.describe().style.background_gradient("summer")


num_cols = train.select_dtypes(include = ['float64', "int64"]).columns.to_list()
sns.heatmap(train[num_cols].corr(), annot = True, fmt = ".2f", cmap = "summer")


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
cat_cols = [i for i in train.columns[1:-1] if i not in num_cols ]
target_col = ['Listening_Time_minutes']


for col in num_cols:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    sns.histplot(train[col], kde=True, ax=ax1, color='skyblue')
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency")
    
    ax2 = ax1.twinx()
    sns.boxplot(x=train[col], ax=ax2, color='salmon', width=0.2)
    ax2.set_ylabel("")

    ax2.set_yticks([])
    plt.title(f'Distribution & Boxplot of {col}',fontsize = 20, color='#001F3F')
    plt.tight_layout()
    plt.show()


train[cat_cols].nunique()


# Podcast_Name Count Plot
plt.figure(figsize = (14, 10))
train["Podcast_Name"].value_counts().plot(
    kind = "barh",
    color=sns.color_palette("husl", n_colors=train["Podcast_Name"].nunique())
)
plt.title("Count Plot of Podcast_Name",fontsize = 20, color = "#001F3F")
plt.show()

# Episode_Title Count Plot
plt.figure(figsize = (14, 14))
train["Episode_Title"].value_counts().plot(
    kind = "barh",
    color=sns.color_palette("husl", n_colors=train["Episode_Title"].nunique())
)
plt.title("Count Plot of Episode_Title",fontsize = 20, color = "#001F3F")
plt.show()


cols = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
for col in cols:
    plt.figure(figsize=(9,6))
    sns.countplot(data=train, x=col, palette = "husl")
    plt.title(f'Count Plot of {col}', color = "#001F3F")
    plt.show()


fig, axes = plt.subplots(1, 2, figsize = (12, 5))
sns.histplot(train[target_col], ax = axes[0],kde = True, alpha = 0.5)
sns.boxplot(train[target_col], ax = axes[1], color = "lightcoral")
plt.show()


# Podcast Name vs Listening_time_minutes
plt.figure(figsize=(14, 8))
sns.boxplot(
    data=train,
    x="Podcast_Name",
    y="Listening_Time_minutes",
    palette="Set2"
)
plt.xticks(rotation=90)
plt.title("Listening Time by Podcast", color="#001F3F")
plt.show()


cols = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Number_of_Ads"]
for col in cols:
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=train,
        x=col,
        y="Listening_Time_minutes",
        palette="Set2"
    )
    plt.title(f"Listening Time by {col}", color="#001F3F")
    plt.show()


sns.scatterplot(
    data = train,
    x = "Episode_Length_minutes",
    y = "Listening_Time_minutes",
    color = "teal"
)
plt.title("Episode Length minutes vs Listening Time Minutes", color = "#001F3F")
plt.show()


sns.scatterplot(
    data = train,
    x = "Host_Popularity_percentage",
    y = "Listening_Time_minutes",
    color = "teal"
)
plt.title("Host Popularity Percentage vs Listening Time Minutes", color = "#001F3F")
plt.show()

sns.scatterplot(
    data = train,
    x = "Guest_Popularity_percentage",
    y = "Listening_Time_minutes",
    color = "teal"
)
plt.title("Guest Popularity Percentage vs Listening Time Minutes", color = "#001F3F")
plt.show()


# Filling nan values in Number_of_Ads with mode value as there is only 1 null value
train["Number_of_Ads"] = train["Number_of_Ads"].fillna(train["Number_of_Ads"].mode()[0])
train["Number_of_Ads"].isna().sum()


train["Episode_Length_minutes"] = train["Episode_Length_minutes"].fillna(train["Episode_Length_minutes"].median())
test["Episode_Length_minutes"] = test["Episode_Length_minutes"].fillna(test["Episode_Length_minutes"].median())

train["Guest_Popularity_percentage"] = train["Guest_Popularity_percentage"].fillna(train["Guest_Popularity_percentage"].median())
test["Guest_Popularity_percentage"] = test["Guest_Popularity_percentage"].fillna(test["Guest_Popularity_percentage"].median())

train.isnull().sum().sum(), test.isnull().sum().sum()


# Identifying Outliers
q1 = np.percentile(train["Episode_Length_minutes"], 25)
q3 = np.percentile(train["Episode_Length_minutes"], 75)
iqr = q3 - q1
mini = q1 - 1.5 * iqr
maxi = q3 + 1.5 * iqr

outliers_mask = (train["Episode_Length_minutes"] < mini) | (train["Episode_Length_minutes"] > maxi)
train[outliers_mask]

outliers_mask = (test["Episode_Length_minutes"] < mini) | (test["Episode_Length_minutes"] > maxi)
test[outliers_mask]


# Capping Values
train["Episode_Length_minutes"] = train["Episode_Length_minutes"].clip(lower=mini, upper=maxi)
test["Episode_Length_minutes"] = test["Episode_Length_minutes"].clip(lower=mini, upper=maxi)


# Number of Ads
train["Number_of_Ads"].value_counts()


# If the number of ads is more than 3 then replace with mode 0
train["Number_of_Ads"] = train["Number_of_Ads"].apply(lambda x:0 if x > 3 else x)
test["Number_of_Ads"] = test["Number_of_Ads"].apply(lambda x:0 if x > 3 else x)


X = train.drop(columns = ["id", "Listening_Time_minutes"])
y = train["Listening_Time_minutes"]
test_ids = test['id']
test = test.drop(columns = "id")


def create_features(train):
    
    # Captures overall combined star power of the host and guest
    train["Host_Guest_Combo_Popularity"] = train["Host_Popularity_percentage"] * train["Guest_Popularity_percentage"]

    # Normalizes ad count by episode length — a measure of ad density
    train["Ads_per_Minute"] = train["Number_of_Ads"] / (train["Episode_Length_minutes"] + 0.00001)
    
    # Measures dominance of host popularity vs guest
    train["Host_to_Guest_Popularity_Ratio"] = train["Host_Popularity_percentage"]/ (train["Guest_Popularity_percentage"] + 0.00001)
    
    # Difference in star power — useful if the gap impacts performance
    train["Popularity_Difference"] = train["Host_Popularity_percentage"] - train["Guest_Popularity_percentage"]
    
    # Captures weighted presence of host/guest by episode duration
    train["Length_x_Host_Popularity"] = train["Episode_Length_minutes"] * train["Host_Popularity_percentage"]
    train["Length_x_Guest_Popularity"] = train["Episode_Length_minutes"] * train["Guest_Popularity_percentage"]
    
    # A rough proxy for how much value the show tries to extract (ads) based on star power
    train["Ad_Density_Popularity"] = train["Number_of_Ads"] * (train["Host_Popularity_percentage"] + train["Guest_Popularity_percentage"])
   
    # Weekend drops might behave differently in engagement
    train["Weekend_Publication"] = train["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)

    return train

X = create_features(X.copy())
test = create_features(test.copy())
X.shape, test.shape


X.select_dtypes(include = ["float64", "int64"]).columns


cols = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

encoder = TargetEncoder()
X[cols] = encoder.fit_transform(X[cols], y)
test[cols] = encoder.transform(test[cols])


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


xgb_model = XGBRegressor()
lgbm_model = LGBMRegressor(verbose = 0)
catboost_model = CatBoostRegressor(verbose=0)


# Train and evaluate models
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model,
    'CatBoost': catboost_model,
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_val_pred, squared=False)
    print(f'{name} RMSE: {rmse}')

# Select the best model based on RMSE
best_model_name = min(models, key=lambda name: mean_squared_error(y_test, models[name].predict(X_test), squared=False))
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')


model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=5,
    loss_function='RMSE',          
    eval_metric='RMSE',             
    early_stopping_rounds=50,      
    verbose=100                    
)
model.fit(X_train, y_train, eval_set = (X_test, y_test))


model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=5,
    loss_function='RMSE',          
    eval_metric='RMSE',             
    early_stopping_rounds=50,      
    verbose=100                    
)
model.fit(X, y)


y_pred_test = model.predict(test)
plt.hist(y_pred_test)
plt.show()


y_pred_test = y_pred_test.reshape(-1, 1)
test_ids = test_ids.values.reshape(-1, 1)
submission_df = pd.DataFrame(np.concatenate((test_ids, y_pred_test), axis = 1), columns = ["id", "Listening_Time_minutes"])
submission_df["id"] = submission_df["id"].astype(np.int32)
submission_df


submission_df.to_csv("submission.csv", index = False)
print("File created successfully")

