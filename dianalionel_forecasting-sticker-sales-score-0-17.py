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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns


train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train.head()


train.store.value_counts()


train['product'].unique()


train['country'].unique()


train.isnull().sum()


#Grouping missing values by store,product,country and imputing grouped mean for missing values 

grouped_means = train.groupby(['store', 'product', 'country'])['num_sold'].transform('mean')

# Fill missing values in the target column with the grouped means
train['target'] = train['num_sold'].fillna(grouped_means)


train.isnull().sum()


#Fill the remaining missing values by grouping product,country and imputing grouped mean for remaining missing values.

train['target'] = train['target'].fillna(
    train.groupby(['product', 'country'])['target'].transform('mean')
)



train.isnull().sum()


df=train.copy()


#Converting Date to Day, Month,Year, Weekend

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['date'])

# Extract components
df['DayName'] = df['Date'].dt.day_name()            
df['MonthName'] = df['Date'].dt.month_name()        
df['Year'] = df['Date'].dt.year                     
df['Weekend/Weekday'] = df['Date'].dt.dayofweek     

# Map dayofweek to Weekend/Weekday
df['Weekend/Weekday'] = df['Weekend/Weekday'].map(lambda x: 'Weekend' if x >= 5 else 'Weekday')



df.head()


from sklearn.preprocessing import LabelEncoder

# Specify the columns to encode
columns_to_encode = ['country', 'store', 'product']

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Perform label encoding for specific columns
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the updated DataFrame
print(df)



df.head()


df_train=df.copy()


from sklearn.preprocessing import LabelEncoder

# Specify the columns to encode
columns_to_encode = ['DayName', 'MonthName', 'Weekend/Weekday']

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Perform label encoding for specific columns
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Display the updated DataFrame
print(df)



df2=df.drop(columns=['id', 'date','num_sold','Date','DayName','MonthName','Year'])



# Creating Separate features using multiple combinations

df2['product_country'] = df2['product'] * 10 + df2['country']
df2['store_country'] = df2['store'] * 10 + df2['country']
df2['store_product'] = df2['store'] * 10 + df2['product']



# Define features (X) and target (y)
X = df2.drop(columns=['target'])  # Replace 'target_column' with the name of your target variable
y = df2['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)




# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(mse)


y_pred


y_test


test.head()


# Convert the 'Date' column to datetime format
test['Date'] = pd.to_datetime(test['date'])

# Extract components
test['DayName'] = test['Date'].dt.day_name()            
test['MonthName'] = test['Date'].dt.month_name()        
test['Year'] = test['Date'].dt.year                    
test['Weekend/Weekday'] = test['Date'].dt.dayofweek     

# Map dayofweek to Weekend/Weekday
test['Weekend/Weekday'] = test['Weekend/Weekday'].map(lambda x: 'Weekend' if x >= 5 else 'Weekday')




df_test=test


from sklearn.preprocessing import LabelEncoder

# Specify the columns to encode
columns_to_encode = ['country', 'store', 'product','DayName','MonthName','Year','Weekend/Weekday']

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Perform label encoding for specific columns
for column in columns_to_encode:
    df_test[column] = label_encoder.fit_transform(df_test[column])

# Display the updated DataFrame
print(df_test)


df_test['product_country'] = df_test['product'] * 10 + df_test['country']
df_test['store_country'] = df_test['store'] * 10 + df_test['country']
df_test['store_product'] = df_test['store'] * 10 + df_test['product']


df_test=df_test.drop(['id','date','Date','DayName','MonthName','Year'],axis=1)



pred = model.predict(df_test)


pred


df_test.head()


num_sold = pred


num_sold


submission = pd.DataFrame()
submission['id']=test['id']
submission['num_sold']= num_sold



submission.head()


submission.to_csv('submission.csv', index=False)


# Get feature importances
feature_importances = model.feature_importances_

# Use the actual column names from your dataset
feature_names = X.columns  # Replace `X` with your feature DataFrame

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# Compute correlation matrix
correlation_matrix = df2.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(
    correlation_matrix, 
    annot=True,  # Display the correlation values
    cmap='coolwarm',  # Choose a color map
    fmt=".2f",  # Format the correlation values to 2 decimal places
    linewidths=0.5  # Add gridlines between cells
)
plt.title('Correlation Heatmap')
plt.show()


column_x = 'product'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()



column_x = 'country'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()



column_x = 'store'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()



column_x = 'Year'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=df, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()


column_x = 'DayName'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=df_train, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()


column_x = 'Weekend/Weekday'
column_y = 'target'

sns.barplot(x=column_x, y=column_y, data=df_train, color='blue')
plt.title(f'Bar Chart of {column_x} vs {column_y}')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.show()


train['country.store'] = train['country'] + "_" + train['store']
train['country.product'] = train['country'] + "_" + train['product']
train['store.product'] = train['store'] + "_" + train['product']


import seaborn as sns
import matplotlib.pyplot as plt

column_x = 'target'
column_y = 'country.store'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_y} vs {column_x}')
plt.xlabel(column_x)
plt.ylabel(column_y)



import seaborn as sns
import matplotlib.pyplot as plt


column_x = 'target'
column_y = 'country.product'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_y} vs {column_x}')
plt.xlabel(column_x)
plt.ylabel(column_y)



import seaborn as sns
import matplotlib.pyplot as plt


column_x = 'target'
column_y = 'store.product'

sns.barplot(x=column_x, y=column_y, data=train, color='blue')
plt.title(f'Bar Chart of {column_y} vs {column_x}')
plt.xlabel(column_x)
plt.ylabel(column_y)

