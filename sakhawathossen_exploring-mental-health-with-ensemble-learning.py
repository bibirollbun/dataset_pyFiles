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


#Core Libraries 
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

#Visualization Libraries 

import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning Libraries 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder ,FunctionTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier ,HistGradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,IsolationForest
from sklearn.compose import ColumnTransformer




df_train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')



df_train.head()


df_train.shape


df_train.isnull().sum()


# there have a missing value both numerical and categorical col
# detect the both col with the help of  datatype


df_train.dtypes


# here remove the useless two col id or target col which is depression


df_train = df_train.drop(['id'],axis = 1)





# Select the categorical and numerical columns


categorical = df_train.select_dtypes(include = ['object']).columns

numerical = df_train.select_dtypes(include = ['float64']).columns


categorical


numerical


categorical.isnull().sum()


df_train['Name'].value_counts()


df_train['City'].value_counts()


#here using a loop and find the unique or duplicate  value in col



for col in categorical:
    print(df_train[col].value_counts().head(10))
    print("_" *40)


for col in categorical:
    print(f"Missing Values : {df_train[col].isnull().sum()}")
    print("_" *40)




for col in categorical:
    Missing_values =df_train[col].isnull().sum()
    missing_percentage =(Missing_values/len(df_train))*100
    print(f"Column: {col} ,missing val : {Missing_values} , missing % : {missing_percentage}")
    print("_" *40)

    if missing_percentage > 80 :
        df_train.drop(columns = [col] ,inplace = True)
    elif missing_percentage <30 :
        mode_val = df_train[col].mode()[0]
        df_train[col].fillna(mode_val,inplace = True)
        



# After hanlading missing data


for col in categorical:
    print(f"Missing Values : {df_train[col].isnull().sum()}")
    print("_" *40)




# handaling numerical col


for col in numerical:
    print(df_train[col].value_counts().head(10))
    print("_" *40)


for col in numerical:
    print(f"Missing Values : {df_train[col].isnull().sum()}")
    print("_" *40)




for col in numerical:
    Missing_values =df_train[col].isnull().sum()
    missing_percentage =(Missing_values/len(df_train))*100
    print(f"Column: {col} ,missing val : {Missing_values} , missing % : {missing_percentage}")
    print("_" *40)


print("The skewness of columns:")
print(df_train[numerical].skew())


# A skewness value between -0.5 and 0.5 suggests a roughly normal distribution—your values fit this range.
# Negative skew (left-skewed): Age, Academic Pressure, CGPA, Work/Study Hours—These distributions may have a slight tail on the left.
# Positive skew (right-skewed): Job Satisfaction, Financial Stress—A slight tail on the right.
# Near-zero skewness: Work Pressure, Study Satisfaction—These are almost perfectly symmetric, which is great.



for col in numerical:
    sns.histplot(df_train[col],kde = True ,bins =10 ,label =col )
    plt.xlabel("values")
    plt.ylabel("Frequency")
    plt.title("skewness of numerical columns")
    plt.legend()
    plt.show()



sns.histplot(df_train['Age'],kde = True ,bins = 30 ,label =col )


# Define threshold
threshold = 50  # Percentage threshold for dropping columns

# Calculate skewness for numerical features
skewness_values = df_train[numerical].skew()

# Handling missing values
for col in numerical:
    Missing_values = df_train[col].isnull().sum()
    missing_percentage = (Missing_values / len(df_train)) * 100
    skew_value = skewness_values[col]

    print(f"Column: {col} | Missing Values: {Missing_values} | Missing %: {missing_percentage:.2f}% | Skewness: {skew_value:.2f}")
    print("_" * 40)

    # Drop columns with excessive missing values (>50%)
    if missing_percentage > threshold:
        df_train.drop(columns=[col], inplace=True)
        print(f"Dropped column '{col}' due to excessive missing values.")

    # Handle missing values based on distribution
    else:
        if -0.5 <= skew_value <= 0.5:  # Normally distributed
            df_train[col].fillna(df_train[col].mean(), inplace=True)
            print(f"Filled missing values in '{col}' using Mean.")

        elif skew_value > 0.5 or skew_value < -0.5:  # Skewed distribution
            df_train[col].fillna(df_train[col].median(), inplace=True)
            print(f"Filled missing values in '{col}' using Median.")

print("\nUpdated DataFrame:")
print(df_train.head())


df_train


# using heatmap find the corr matrix


sns.heatmap(df_train.corr(numeric_only = True) ,annot = True)


# FIND THE IMPORTANT FEATURES


correlation_matrix = df_train.corr(numeric_only=True)

# Extract correlation values for Depression
target_correlation = correlation_matrix["Depression"].sort_values(ascending=False)
# Display correlation values
print("Feature correlation with Depression:")
print(target_correlation)





# Visualizing correlation using bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlation.index, y=target_correlation.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.ylabel("Correlation Coefficient")
plt.title("Feature Correlation with Depression")
plt.show()



# FIND THE IMPT FEATURES IN CATEGORICAL DATA USING R MODEL


# define the features and trget
X = df_train [categorical]
df_train['Depression'] =  df_train['Depression'] .replace({0:"NO",1:"Yes"})
y = df_train['Depression']


X





y


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


# Apply One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(df_train[categorical]))

# Assign proper column names
X_encoded.columns = encoder.get_feature_names_out(categorical)

# Concatenate encoded features with original DataFrame
df_train_encoded = pd.concat([df_train, X_encoded], axis=1).drop(columns=categorical)

# Define target variable (assuming Depression is numerical)
y = df_train_encoded["Depression"]

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(df_train_encoded.drop(columns=["Depression"]), y)

# Feature importance analysis
importance_values = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": df_train_encoded.drop(columns=["Depression"]).columns,
                                      "Importance": importance_values}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance using Random Forest:")
print(feature_importance_df)



# Define features and target
X_train = df_train.drop('Depression', axis=1)
y_train = df_train['Depression']


# Redefine columns for preprocessing after feature engineering
numerical_col = X_train.select_dtypes(include = ['float64', 'int64']).columns.tolist()
categorical_col = X_train.select_dtypes(include = ['object']).columns.tolist()




# processing pipeline 

numerical_pipeline = Pipeline(
    steps = [
        ('imputer',SimpleImputer(strategy ='median')),
        ('scaler' , StandardScaler()),
        ('convert_to_float32' , FunctionTransformer(lambda x: x.astype(np.float32)))
    ]
)

categorical_pipeline = Pipeline(
    steps = [
        ('imputer',SimpleImputer(strategy ='constant',fill_value = 'missing')),
        ('ordinal' , OrdinalEncoder(dtype = np.int32,handle_unknown = 'use_encoded_value',unknown_value = -1))
    ]
)

# Combine the numerical and categorical pipelines

preprocessor = ColumnTransformer(
    transformers = [
        ('num',numerical_pipeline,numerical_col),
        ('cat',categorical_pipeline,categorical_col)
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)


X_train_preprocessed


# outlier detection on the training data


# 1.Box Plot

plt.figure(figsize = (10,6))
sns.boxplot(data =X_train_preprocessed)
plt.title("Box plot for outlier Detection")
plt.show()


# 2.Histogram 

plt.figure(figsize = (10,6))
plt.hist(X_train_preprocessed.flatten() ,bins= 50 , edgecolor = 'black')
plt.title("Histogram for outlier Detection")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# 3.Isolation Forest Decision Score PLot

isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(X_train_preprocessed)


scores = isolation_forest.decision_function(X_train_preprocessed)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


#  Isolation Forest decision score histogram follows a normal distribution, it generally means the dataset does not contain extreme outliers or anomalies. This can indicate that most points in your dataset are behaving consistently, without significant deviations.



# define the outlier label -1 or 1 . if there will be outlier its 1 otherwise -1
outlier_label = isolation_forest.fit_predict(X_train_preprocessed)


outlier_label


non_outlier = outlier_label!=-1
non_outlier.sum()


have_outlier = outlier_label==-1
have_outlier.sum()


# Filter out outlier from X_train_preprocessed and y_train tata


X_train_preprocessed = X_train_preprocessed[non_outlier]
y_train = y_train[non_outlier]


# After clean outlier Isolation Forest Decision Score PLot 

isolation_forest  = IsolationForest(contamination = 0.04 ,random_state =42)
isolation_forest.fit(X_train_preprocessed)


scores = isolation_forest.decision_function(X_train_preprocessed)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor='black')
plt.title("Isolation Forest Decision Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


# Define parameters
xgb_params = {
     'learning_rate': 0.298913248058474, 
     'max_depth': 9, 
     'min_child_weight': 3, 
     'n_estimators': 673, 
     'subsample': 0.5933970249700855, 
     'gamma': 2.597137534750985, 
     'reg_lambda': 0.11328048420927406, 
     'colsample_bytree': 0.1381203919800721
}

catboost_params = {
    'iterations': 145, 
    'depth': 7, 
    'learning_rate': 0.29930179265937246, 
    'l2_leaf_reg': 1.242352421942431, 
    'random_strength': 8.325681754379957, 
    'bagging_temperature': 0.7869848919618048, 
    'border_count': 139
}

hgb_params = {
    'learning_rate': 0.16299202834206894, 
    'max_iter': 250, 
    'max_depth': 4, 
    'l2_regularization': 7.1826466833939895,
    'early_stopping': True
}

# Initialize models with pre-tuned and trial-specific parameters
xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, random_state=42)
catboost_model = CatBoostClassifier(**catboost_params, random_state=42, verbose=0)
hgb_model = HistGradientBoostingClassifier(**hgb_params, random_state=42)

# Define stacking ensemble with the LightGBM model tuned in this trial
stacking_ensemble = StackingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgb', xgb_model),
        ('hgb', hgb_model)
    ],
    final_estimator=LogisticRegression(),
    passthrough=False
)


# Define a scoring metric
scoring = make_scorer(accuracy_score)

# Perform cross-validation
cv_scores = cross_val_score(stacking_ensemble, X_train_preprocessed, y_train, cv=5, scoring=scoring)

# Print cross-validation results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")


# 6. Test Result




