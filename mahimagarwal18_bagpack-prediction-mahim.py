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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv').set_index('id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv').set_index('id')
train_extra_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv').set_index('id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_data.sample(5)


train_data.shape


train_data.isnull().sum()


train_data = pd.concat([train_data, train_extra_data], ignore_index=True)


train_data.shape


test_data.shape


train_data.isnull().sum()


# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(30, 28))  # Initialize figure
# plots = 1  # Track subplot position

# for col in train_data.columns:
#     if train_data[col].dtype == object:  # Check if column is categorical
#         plt.subplot(4, 4, plots)  # Set subplot position for boxplot
        
#         # Fix: Assign hue to avoid warning
#         sns.boxplot(data=train_data, x=col, y='Price', hue=col, palette='pastel', dodge=False)
#         plt.xlabel(col, fontsize=28)
#         plt.ylabel('Price',fontsize=28)
#         plt.title(f'Boxplot for {col} and its Effect on Price')
#         plt.xticks(rotation=45,fontsize=18)
#         plt.yticks(fontsize=18)
#         plots += 1  # Move to the next subplot
        
#         # Add a pie chart for the same categorical column
#         plt.subplot(4, 4, plots)  # Set subplot position for pie chart
#         train_data[col].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel1', textprops={'fontsize': 14})
#         plt.ylabel('')  # Hide y-label to keep it clean
#         plt.title(f'Category Distribution of {col}')
#         plots += 1  # Move to the next subplot

# plt.tight_layout()


train_data['Compartments'] /= 2
train_data['Weight Capacity (kg)'] /= 2


test_data['Compartments'] /= 2
test_data['Weight Capacity (kg)'] /= 2


categorical_cols = ['Brand','Material','Size','Laptop Compartment',	'Waterproof','Style','Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']


for col in categorical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

for col in numerical_cols:
    train_data[col].fillna(train_data[col].median())
    test_data[col].fillna(test_data[col].median())
    


train_data['Weight Capacity (kg)'] = train_data['Weight Capacity (kg)'].fillna(train_data['Weight Capacity (kg)'].mean())
test_data['Weight Capacity (kg)'] = test_data['Weight Capacity (kg)'].fillna(test_data['Weight Capacity (kg)'].mean())


train_data.isnull().sum()


test_data.isnull().sum()


from sklearn.preprocessing import LabelEncoder, StandardScaler

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


for col in numerical_cols:
    scaler = StandardScaler()
    train_data[col] = scaler.fit_transform(train_data[[col]]) 
    test_data[col] = scaler.transform(test_data[[col]])   


train_data.sample(5)


test_data.sample(5)


from sklearn.model_selection import train_test_split

X = train_data.drop(columns=['Price'])
y = train_data['Price']


print(X.isnull().sum())  # Check missing values column-wise
print('-------------------------------------')
print('-------------------------------------')
print(y.isnull().sum())  # Check if target variable has NaNs



from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)




cvs_lr = cross_val_score(lr_model, X_val, y_val, cv=3, scoring='neg_mean_squared_error')


rmse = np.sqrt(-cvs_lr)
print(rmse)





predictions = lr_model.predict(test_data)

submission['Price'] = predictions
submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('submission.csv')

submission




