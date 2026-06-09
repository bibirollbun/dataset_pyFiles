# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



print(f"Dataset Shape: {df_train.shape}")

print("\nData Info:")
df_train.info()

print("\nNumerical Features Summary:")
display(df_train.describe())


print("\nFirst 10 Rows of the Dataset:")
display(df_train.head(10))


num_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Listening_Time_minutes",
]

for feature in num_features:
    plt.figure(figsize=(10,4))
    # hist
    plt.subplot(1,2,1)
    sns.histplot(df_train[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    #boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_train[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"Number of Missing Values: {df_train[feature].isnull().sum()}")


cat_features = [
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment"
]

for feature in cat_features:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df_train[feature], order=df_train[feature].value_counts().index)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Number of Unique {feature}: {df_train[feature].unique()}")
    print(f"Missing Values in {feature}: {df_train[feature].isnull().sum()}")


for feature in num_features:
        plt.figure(figsize=(10,4))
        sns.scatterplot(x=df_train[feature],y = df_train["Listening_Time_minutes"])
        plt.title(f"{feature} vs Listening_Time_minutes")
        plt.xlabel(f"{feature}")
        plt.ylabel("Listening_Time_minutes")
        plt.show()

#correlation matrix
corr_matrix = df_train[num_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


for feature in cat_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_train[feature], y=df_train["Listening_Time_minutes"])
        plt.title(f"{feature} vs. Listening_Time_minutes")
        plt.xlabel(feature)
        plt.ylabel("Listening_Time_minutes")
        plt.xticks(rotation=45)
        plt.show()


df_test.info()



print("Missing Values per Column:")
print(df_train.isnull().sum())

num_features1 = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]

#strategy as median 
num_imputer = SimpleImputer(strategy="median")
df_train[num_features1] = num_imputer.fit_transform(
    df_train[num_features1]
)
df_test[num_features1]=num_imputer.transform(df_test[num_features1])

# strategy as most frequent value
cat_imputer = SimpleImputer(strategy="most_frequent")
df_train[cat_features] = cat_imputer.fit_transform(
    df_train[cat_features]
)
df_test[cat_features]=cat_imputer.transform(df_test[cat_features])

df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()


print("\nMissing Values After Imputation:")
print(df_train.isnull().sum())



from sklearn.preprocessing import OneHotEncoder
cat_features1 = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment"  
]

ohe = OneHotEncoder(drop=None,sparse=False, handle_unknown='ignore')

encoded_cat_train = pd.DataFrame(
    ohe.fit_transform(df_train[cat_features1]),
    columns=ohe.get_feature_names_out(cat_features1),
    index=df_train.index
)

encoded_cat_test = pd.DataFrame(
    ohe.transform(df_test[cat_features1]),
    columns=ohe.get_feature_names_out(cat_features1),
    index=df_test.index
)

df_train = df_train.drop(columns=cat_features1)
df_test = df_test.drop(columns=cat_features1)


df_train = pd.concat([df_train, encoded_cat_train], axis=1)
df_test = pd.concat([df_test, encoded_cat_test], axis=1)


X_train= df_train.drop(columns=["Listening_Time_minutes"])
y_train= df_train["Listening_Time_minutes"]

X_kaggletest = df_test


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)


param_dist = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'min_child_weight': randint(1, 10),
}


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)


random_search.fit(X_train, y_train)


print("Best hyperparameters:", random_search.best_params_)

y_pred = random_search.predict(X_kaggletest)


print(df_sub.head(10))


df_sub['Listening_Time_minutes'] = y_pred
df_sub.to_csv('submission.csv', index=False)


# Saving the submission file
submission = pd.DataFrame({'id': df_sub['id'], 'Listening_Time_minutes': y_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
import os
print(os.listdir('/kaggle/working/'))

