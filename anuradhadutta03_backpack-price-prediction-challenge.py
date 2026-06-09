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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")




test.head().T


train.head().T


train_extra.head().T


train.isnull().sum()


train.describe(include="all").T


train.info()


train.duplicated().sum()


train.nunique()


shape_rows=train.shape[0]
print("Number of rows =",shape_rows)


shape_columns=train.shape[1]
print("Number of columns =",shape_columns)


categorical_columns_train = [
    "Brand",
    "Material",
    "Size",
    "Compartments",
    "Laptop Compartment",
    "Waterproof",
    "Style",
    "Color",
]

numerical_columns_train = ['Weight Capacity (kg)', 'Price']
    


train['Brand'].value_counts().plot(kind='bar')


sns.pairplot(train[['Price']])
plt.show()



train['Material'].value_counts().plot(kind='bar')


sns.pairplot(train[['Price']])




# Sample data (Replace with your actual dataset)
data = train['Price']

# Set the size of the plot
plt.figure(figsize=(6, 6))

# Create the histogram
sns.histplot(data, bins=30, kde=True, color='skyblue', edgecolor='black', alpha=0.7)

# Add title and labels
plt.title('Price Distribution', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Customize the grid
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()



sns.boxplot(x='Brand', y='Price', data=train)
plt.show()



sns.boxplot(x='Material', y='Price', data=train)
plt.show()



plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Price'], color='lightblue')
plt.title('Price Distribution - Boxplot', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.show()



train['Price'].describe()



categorical_columns_train = [
    "Brand",
    "Material",
    "Size",
    "Compartments",
    "Laptop Compartment",
    "Waterproof",
    "Style",
    "Color",
]

numerical_columns_train = ['Weight Capacity (kg)', 'Price']


plt.figure(figsize=(10,8))
for i ,col in enumerate(categorical_columns_train ,1):
    plt.subplot(3,3,i)
    sns.countplot(y=col, data=train)
    plt.title(col)
    
plt.tight_layout()
plt.show()


# Fill missing values for numerical columns with median
train[numerical_columns_train] = train[numerical_columns_train].fillna(train[numerical_columns_train].median())

# Fill missing values for categorical columns with mode
train[categorical_columns_train] = train[categorical_columns_train].fillna(train[categorical_columns_train].mode().iloc[0])



train["Weight Capacity (kg)"].isnull().sum()


def fill_na(df):
    df['Brand'] = df['Brand'].fillna('Unknown')
    df['Material'] = df['Material'].fillna('Unknown')
    df['Size'] = df['Size'].fillna('Unknown')
    df['Laptop Compartment'] = df['Laptop Compartment'].fillna('Unknown')
    df['Waterproof'] = df['Waterproof'].fillna('Unknown')
    df['Style'] = df['Style'].fillna('Unknown')
    df['Color'] = df['Color'].fillna('Unknown')
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())
    return df


train=fill_na(train)
test=fill_na(test)


fill_na(train)


print(train.isnull().sum())



print(test.isnull().sum())


from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Check the columns in the train dataset
print(train.columns)

# Apply One-Hot Encoding to all categorical columns
# Ensure that 'train' has no string columns like 'Nike' or 'Puma' remaining
train = pd.get_dummies(train, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train.drop('Price', axis=1), train['Price'], test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate and print the Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
print(np.sqrt(mse))



# Assuming `test` DataFrame is already loaded and prepared (same encoding as train)
# Apply the same One-Hot Encoding to the test data
test_encoded = pd.get_dummies(test, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])

# Align the test data with the train data columns after encoding (ensure columns match)
test_encoded = test_encoded.reindex(columns=x_train.columns, fill_value=0)

# Extract 'id' from the test set (assuming it has 'id' column)
test_id = test['id']

# Get predictions from the model on the test data
test_predictions = model.predict(test_encoded)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'id': test_id,  # Include the 'id' from the test set
    'Price': test_predictions  # Predicted prices from the model
})

# Save the submission DataFrame to a CSV file
submission.to_csv("submission.csv", index=False)

# Print the first few rows of the submission to check the format
print(submission.head())





