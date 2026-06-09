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


#ignore
import warnings
warnings.filterwarnings("ignore")

#data visualization Libraries 
import matplotlib.pyplot as plt
import seaborn as sns 


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

datafile = {
    "sample_submission": submission,
    "train":train,
    "test": test
}

for name , df in datafile.items():
    print(f"__{name}__")
    print(f"shape of data : {df.shape}")
    print(f"columns : {df.columns}")
    display(df.head())
    print("\n" + "_"*42)



print("SOME BASIC INFO ABOUT THE DATASETS ")
for name , df in datafile.items():
    print(f"__{name}__")
    print(df.info())
    print("\n" + "_"*42)


# Checking for Missing values 
for name , df in datafile.items():
    print(f"__{name}__")
    for column in df.columns:
        null_values = df[column].isna().sum()
        
        print(f" Missing values in {column} : {null_values} = {null_values*100/df.shape[0]}%")
    print("\n" + "_"*42)


#Duplicate rows 
for name , df in datafile.items():
    print(f"__{name}__")
    print(f"Number Duplicate values: {df.duplicated().sum()} = {df.duplicated().sum()*100/df.shape[0]}%")
    print("\n" + "_"*42)


# unique value in each columns 
print("NUMBER OF UNIQUE VALUES IN EVERY FEATURE")
for name , df in datafile.items():
    print(f"__{name}__")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()}")
    print("\n" + "_"*42)


# Numerical Features Summary:
for name , df in datafile.items():
    print(f"__{name}__")
    display(df.describe())
    print("\n" + "_"*42)



numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Listening_Time_minutes",
]

# Plot histograms and box plots for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(12, 4))

    # Histogram 
    plt.subplot(1, 2, 1)
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Box plot to identify outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train[feature].skew():.2f}")
    print(f"Number of Missing Values: {train[feature].isnull().sum()}")
    print("\n" + "__"*42)


numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]

# Plot histograms and box plots for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(12, 4))

    # Histogram 
    plt.subplot(1, 2, 1)
    sns.histplot(test[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Box plot to identify outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=test[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()
        # Print additional statistics
    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train[feature].skew():.2f}")
    print(f"Number of Missing Values: {train[feature].isnull().sum()}")
    print("\n" + "__"*42)


categorical_features = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
]

# Plot bar charts for each categorical feature
for feature in categorical_features:
    plt.figure(figsize=(8, 5))

    if feature in ["Podcast_Name", "Episode_Title"]:
        # For features with many unique values, plot top 10 categories
        top_categories = train[feature].value_counts().nlargest(10)
        sns.barplot(x=top_categories.index, y=top_categories.values)
        plt.title(f"Top 10 {feature} Categories")
    else:
        # For features with fewer categories, plot all
        sns.countplot(
            x=train[feature], order=train[feature].value_counts().index
        )
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.show()

    # Print the number of unique values
    print(f"Number of Unique {feature}: {train[feature].nunique()}")
    print(f"Missing Values in {feature}: {train[feature].isnull().sum()}")
    print("\n" + "__"*42)


# variation of Label wrt numerical features 
numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]

for i , feature in enumerate(numerical_features):
    plt.figure(figsize = (6, 16))
    plt.subplot(4, 1, i+1)
    plt.scatter(train[feature] , train["Listening_Time_minutes"])
    plt.xlabel(feature)
    plt.show()



numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]



from sklearn.impute import SimpleImputer


# Impute numerical features with median
num_imputer = SimpleImputer(strategy="median")
train[numerical_features] = num_imputer.fit_transform(
    train[numerical_features]
)


# Same with Test Data 
test[numerical_features] = num_imputer.fit_transform(
    test[numerical_features]
)

cat_imputer = SimpleImputer(strategy="most_frequent")
test[categorical_features] = cat_imputer.fit_transform(
    test[categorical_features]
)

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print("Train\n")
print(train.isnull().sum())
print("\n"*2)
print("Test\n")
print(test.isnull().sum())



# handling outliers: I am capping it as test data also has outlier.
# IQR

train_new = train.copy()

def handleOutlier(df , column):
    per25 = df[column].quantile(.25)
    per75 = df[column].quantile(.75)
    Iqr = per75 - per25 
    
    upper_limit = per75 + 1.5*Iqr
    lower_limit = per25 - 1.5*Iqr 
    
    df[column] = np.where(
    df[column] > upper_limit,
    upper_limit,
    np.where(
        df[column] < lower_limit,
        lower_limit,
        df[column]
    )
)

handleOutlier(train_new , "Episode_Length_minutes")

    
    


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.histplot(train['Episode_Length_minutes'], kde=True)
plt.title("Histogram Before")

plt.subplot(2,2,2)
sns.boxplot(data = train['Episode_Length_minutes'])
plt.title("BoxPlot Before")

plt.subplot(2,2,3)
sns.histplot(train_new['Episode_Length_minutes'], kde=True)
plt.title("Histogram After")

plt.subplot(2,2,4)
sns.boxplot(data = train_new['Episode_Length_minutes'])
plt.title("BoxPlot Before")

plt.show()


handleOutlier(train_new ,"Number_of_Ads")


plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
sns.histplot(train['Number_of_Ads'], kde=True)
plt.title("Histogram Before")

plt.subplot(2,2,2)
sns.boxplot(data = train['Number_of_Ads'])
plt.title("BoxPlot Before")

plt.subplot(2,2,3)
sns.histplot(train_new['Number_of_Ads'], kde=True)
plt.title("Histogram After")

plt.subplot(2,2,4)
sns.boxplot(data = train_new['Number_of_Ads'])

plt.show()


test_new = test.copy()
handleOutlier(test_new , "Episode_Length_minutes")
handleOutlier(test_new , 'Number_of_Ads')


train_new.dtypes 


from sklearn.preprocessing import LabelEncoder

categorical_cols = train_new.select_dtypes(include=['object']).columns
label_encoders = {} 
for col in categorical_cols:
    le = LabelEncoder()
    train_new[col] = le.fit_transform(train_new[col])
    test_new[col] = le.transform(test_new[col]) 
    label_encoders[col] = le  
train_new = train_new.astype(float)
test_new = test_new.astype(float)

print("Categorical columns converted to numerical successfully!")


train_new.dtypes


test_new.dtypes


train_new = train_new.drop(columns = ["id"])
test_new = test_new.drop(columns = ["id"])


X_train = train_new.drop(columns = ["Listening_Time_minutes"])
y_train = train_new["Listening_Time_minutes"]
X_test = test_new


print(y_train.shape)
print(X_test.shape)
print(X_train.shape)


from sklearn.preprocessing import StandardScaler

se = StandardScaler()
X_train_scaled = se.fit_transform(X_train)
X_test_scaled = se.transform(X_test)


from sklearn.model_selection import train_test_split as tts

X_train , X_val , y_train , y_val = tts(X_train_scaled , y_train , test_size = .2 , random_state = 42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
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

model = XGBRegressor(**xgb_params)
model.fit(X_train , y_train , eval_set=[(X_val, y_val)])
val_pred = model.predict(X_val)
score = rmse(y_val, val_pred)
score


