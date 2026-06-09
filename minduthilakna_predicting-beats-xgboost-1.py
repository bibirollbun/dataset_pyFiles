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


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load train-dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


train.shape


train.head()


# load test-dataset
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


test.shape


test.head()


#Let's explore more in train- dataset
train.info()


train.duplicated().sum()


train.describe()


# numerical features in the data set
numerical_features = ['RhythmScore'	,'AudioLoudness', 'VocalContent',	'AcousticQuality',	'InstrumentalScore', 'LivePerformanceLikelihood',	'MoodScore','TrackDurationMs','Energy', 'BeatsPerMinute']


# Let's plot box plot for each feature to indentify the outliers in the data set
for feature in numerical_features:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=train[feature])
    plt.title(f"Box plot of {feature}")
    plt.show()



# count the number of outliers in each feature
outlier_count={}
for feature in numerical_features:
    Q1 = train[feature].quantile(0.25)
    Q3 = train[feature].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 -1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = train[feature][(train[feature]<lower_bound)|(train[feature].values>upper_bound)]
    outlier_count[feature] = len(outliers)
outliers_count = pd.DataFrame(outlier_count.items(), columns=['Feature', 'Outlier Count'])
outliers_count


# Let's plot histogram of BPM
plt.figure(figsize=(10,6))
sns.histplot(train['BeatsPerMinute'], bins =30, kde= True, color='blue')
plt. title(" Histogram of Beats Per Minute")
plt.xlabel("Beasts Per Minute")
plt.ylabel("Frequency")
plt.show()


# correlation heatmap
plt.figure(figsize =(10,8))
correlation_matrix = train[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# list the features and their correlation with the target variable
correlation_with_target = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
correlation_with_target


# import necessary libraries for modeling
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# X,y for modeling
X = train.drop(columns=['id','BeatsPerMinute'])
y = train['BeatsPerMinute']


# spit the data into  training and test set
X_train, X_val ,y_train, y_val = train_test_split(X,y, test_size= 0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape,y_val.shape


# standardize the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_val= scalar.transform(X_val)


# parameters for tuning
param_grid = {
    'max_depth': [4,6,8],
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[200,500,1000],
    'subsample':[0.7,0.8,1.0],
    'colsample_bytree':[0.7,0.8,1.0]
}


# Define the model
model = XGBRegressor(objective='reg:squarederror',
                      n_jobs=-1,
                      random_state=42)


#train the model
grid_search = GridSearchCV(estimator=model,param_grid= param_grid,scoring= 'neg_root_mean_squared_error',cv=3,verbose=2)


grid_search.fit(X_train,y_train)


print("Best Params:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)


#evaluate with best model
best_model = grid_search.best_estimator_
y_pred= best_model.predict(X_val)


rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2_score= r2_score(y_val, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2_score}")


#plot feature importance
importance = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features, palette='viridis')
plt.title("Feature Importance")
plt.show()


# with test data set
test_no_id = test.drop(columns=['id'])
test_scaled = scalar.transform(test_no_id)
test_predictions = best_model.predict(test_scaled)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


submission = sample_submission.copy()
submission['BeatsPerMinute'] = test_predictions.round(5)
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

