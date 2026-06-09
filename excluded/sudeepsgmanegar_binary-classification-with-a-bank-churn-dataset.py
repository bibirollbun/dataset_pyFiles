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


df_=pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')
df_.head()


df_test=pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
df_test.head()



df_train=pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
df_train.head()


print(df_train)
print(df_test)
print(df_)


num_train_rows, num_train_columns = df_train.shape

num_test_rows, num_test_columns = df_test.shape

num_submission_rows, num_submission_columns = df_.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Submission Data:")
print(f"Number of Rows: {num_submission_rows}")
print(f"Number of Columns: {num_submission_columns}")



print(f'dtypes:{df_train.info()}')
print(f'dtypes:{df_test.info()}')
df_.info()



df_train.nunique()


df_train.isnull().sum()
df_test.isnull().sum()
df_.isnull().sum()


df_train.isnull().sum()


df_test.isnull().sum()


df_train[df_train.duplicated()]


df_test[df_test.duplicated()]


df_[df_.duplicated()]


df_train.describe()


print(f'Train :{df_train.columns}')
print(f'Test :{df_test.columns}')
print(f':{df_.columns}')


train=df_train


test=df_test


train.head()


test.head()


train.drop(columns=['id','CustomerId','Surname'],inplace=True,axis=1)
test.drop(columns=['id','CustomerId','Surname'],inplace=True,axis=1)


train.head()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns

# Histograms
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
     


train['Geography'].unique()


train['Gender'].unique()


categorical_columns=train.select_dtypes(include=['object'])


for col in categorical_columns:
    counts = train[col].value_counts()
    
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(f"Distribution of {col}")
    plt.show()


train.head()


train.describe()



z=train.corr(numeric_only=True)
z


# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
     



# Create boxplots for each numerical column
plt.figure(figsize=(15, 8))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)  # Arrange plots in a grid of 2 rows and 3 columns
    plt.boxplot(train[col], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()


train['NumOfProducts'].value_counts()


columns_to_cap = ['CreditScore','Age']  # Replace with your columns

# Function to cap and floor outliers
def cap_and_floor_outliers(df, columns):
    for col in columns:
        Q1 = train[col].quantile(0.25)  # 25th percentile
        Q3 = train[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1               # Interquartile range

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply capping and flooring
        train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)

    return train


# Apply the function
train = cap_and_floor_outliers(train, columns_to_cap)


plt.figure(figsize=(15, 8))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)  # Arrange plots in a grid of 2 rows and 3 columns
    plt.boxplot(train[col], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()


train.describe()


train.head()


sns.barplot(x='Geography', y='Balance', data=train)

# Customize the plot (optional)
plt.title('Bar Plot for Categorical Column')
plt.xlabel('Category')
plt.ylabel('Value')

# Show the plot
plt.show()


train.head()


plt.figure(figsize=(10,6))
plt.title('Distribution of Gender')
sns.countplot(data=train,x='Gender')


train['CreditScore_Bins'] = pd.cut(train['CreditScore'], bins=10)

# Count values in each bin
bin_counts = train['CreditScore_Bins'].value_counts().sort_index()

# Visualization: Histogram of Bins
plt.figure(figsize=(15, 6))
bin_counts.plot(kind='bar', width=1, color='skyblue', edgecolor='black')
plt.title('Distribution of Credit Score Bins')
plt.xlabel('Credit Score Range')
plt.ylabel('Count')
plt.xticks(rotation=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


train.head()


train_=train.drop(columns=['CreditScore_Bins'])


train_.head()


credit_scores = train['CreditScore']
balances =train['Balance']

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(credit_scores, balances)

# Customize the plot
plt.xlabel('Credit Score')
plt.ylabel('Balance')
plt.title('Credit Score vs. Balance')
plt.grid(True)
plt.show()


# Bar Plot (Average Balance)
plt.figure(figsize=(8, 5))
avg_balance = train.groupby('HasCrCard')['Balance'].mean().reset_index()
sns.barplot(x='HasCrCard', y='Balance', data=train, palette='coolwarm')
plt.title('Average Balance by HasCreditCard')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.ylabel('Average Balance')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(8, 5))
avg_balance = train_.groupby('Exited')['Balance'].mean().reset_index()
sns.barplot(x='Exited', y='Balance', data=train_, palette='coolwarm')
plt.title('Average Balance by customer exited or not')
plt.xlabel('people exited or not')
plt.ylabel('Average Balance')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


train_.head()


plt.figure(figsize=(10, 6))
sns.histplot(data=train_, x='Age', hue='Exited', kde=False, bins=20, palette='Set2', multiple='dodge')
plt.title('Age Distribution by Exited')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Exited', labels=['Not Exited (0)', 'Exited (1)'])
plt.show()



train_.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


train_.head()


data=train_.copy()


data1=data.copy()



lr=LabelEncoder()
data['Gender']=lr.fit_transform(data['Gender'])



data['Geography']=lr.fit_transform(data['Geography'])


data.head()



X=data.drop(columns=['Exited'])
y=data[['Exited']]


print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))






import lightgbm as lgb
import xgboost as xgb


lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# Evaluate LightGBM
lgb_accuracy = accuracy_score(y_test, lgb_pred)
lgb_report = classification_report(y_test, lgb_pred)
lgb_cm = confusion_matrix(y_test, lgb_pred)

print("LightGBM Accuracy:", lgb_accuracy)
print("LightGBM Classification Report:\n", lgb_report)
print("LightGBM Confusion Matrix:\n", lgb_cm)


data1.head()


data1=data.copy()


data.head()


from sklearn.tree import DecisionTreeClassifier


dt_model = DecisionTreeClassifier(random_state=42) 
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)


# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_report = classification_report(y_test, dt_pred)


print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:\n", dt_report)


xgb_model = xgb.XGBClassifier(random_state=42) 
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)




# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_report = classification_report(y_test, xgb_pred)
xgb_cm = confusion_matrix(y_test, xgb_pred)


print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Classification Report:\n", xgb_report)
print("XGBoost Confusion Matrix:\n", xgb_cm)


import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor  # Use XGBClassifier for classification tasks
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}



grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",  # Change this to 'accuracy' or other metrics for classification
    cv=3,  # Number of cross-validation folds
    verbose=1,  # Increase this for more output during the search
    n_jobs=-1,  # Use all available cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)


# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)

# Test the model with the best parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error:", mse)


