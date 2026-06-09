import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")


train_data.shape


test_data.shape


train_data.head()


train_data.info()


test_data.info()


#check null values
train_data.isnull().sum()


test_data.isnull().sum()


train_data.duplicated().sum()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(18,12))
plt.title("Visualizing Missing Values")
sns.heatmap(train_data.isnull(), cbar=False, yticklabels=False);


# Drop 'id' column in both datasets
train_data = train_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)


categorical_columns = train_data.select_dtypes(include=['object']).columns
numerical_columns = train_data.select_dtypes(exclude=['object']).columns.drop('Depression')


categorical_columns, numerical_columns


# Handle null values in numerical columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

# Apply imputation to relevant columns, which are numerical and have null values
columns_to_impute = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress']
train_data[columns_to_impute] = imputer.fit_transform(train_data[columns_to_impute])


test_copy = test_data.copy() 
test_copy[columns_to_impute] = imputer.fit_transform(test_copy[columns_to_impute])


train_data.isnull().sum()


# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le


label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    test_copy[col] = le.fit_transform(test_copy[col])
    label_encoders[col] = le


train_data.isnull().sum()


# Prepare features and target variable
X = train_data.drop(columns=['Depression'])
y = train_data['Depression']


selected_features = ['Age', 'Financial Stress', 'Academic Pressure', 'Working Professional or Student', 'Work Pressure', 'Have you ever had suicidal thoughts ?', 'Job Satisfaction', 'Work/Study Hours', 'Dietary Habits', 'Sleep Duration']


X_selected = X[selected_features]


# Assuming 'y' is your target variable (Depression)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)


# Train the Random Forest model
from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=10,
    random_state=42
)
my_model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error

#calculate mean absolute error
predictions = my_model.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))


test_copy = test_copy[selected_features]


sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e11/sample_submission.csv")
sample_submission["Depression"] =  my_model.predict(test_copy)
sample_submission["Depression"] = (sample_submission["Depression"] >= 0.5).astype(int)
sample_submission.to_csv("submission.csv",index=False)
sample_submission

