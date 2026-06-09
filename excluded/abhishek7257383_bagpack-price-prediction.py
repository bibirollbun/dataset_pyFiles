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
import seaborn as sns


train_A = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col = "id");
train_B = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col = "id"); 


train_A.info();  #training data info


train_B.info();


train_A.head()


total_trainA_rows =  len(train_A)
print("total training rows are:" ,total_trainA_rows,"\n")
total_trainB_rows = len(train_B)
print("total training extra rows:" ,total_trainB_rows)


missing_trainA_data = train_A.isnull().sum()
print(missing_trainA_data,"\n")
print("total number of missing data in training dataset:",missing_trainA_data.sum())


missing_trainB_data = train_B.isnull().sum()
print(missing_trainB_data,"\n")
print("total number of missing data in extra training dataset:",missing_trainB_data.sum())


train_A.hist(bins=50,figsize=(15,10))
plt.show()


#checking for outliers
sns.boxplot(data=train_A[['Compartments', 'Weight Capacity (kg)', 'Price']])
plt.show() 


def optimize_dataframe(df): 
    if "Compartments" in df.columns: 
        df['Compartments'] = df['Compartments'].astype(int)

    object_columns = df.select_dtypes(include="object").columns  
    df[object_columns] = df[object_columns].astype("category")
 
    return df  


train_A = optimize_dataframe(train_A)
print(train_A.dtypes)


train_A["Brand"].unique()


train_A["Material"].unique()


train_A["Size"].unique()


train_A["Laptop Compartment"].unique()


train_A["Waterproof"].unique()


train_A["Style"].unique()


train_A["Color"].unique()


train_A.describe()


def fill_nulls(df, columns):
    for column in columns:
        value = df[column].mode()[0]
        df[column] = df[column].fillna(value)
        print(value,"\n")
    return df   


train_A = fill_nulls(train_A, ['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color'])


value = train_A['Weight Capacity (kg)'].mean()
print(value)
train_A['Weight Capacity (kg)'] = train_A['Weight Capacity (kg)'].fillna(value)


new_trainA_data = train_A.isnull().sum()
print(new_trainA_data,"\n")


size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}  
train_A['Size'] = train_A['Size'].map(size_mapping) 
train_A = pd.get_dummies(train_A, columns=['Brand', 'Material', 'Style', 'Color'], drop_first=True) 
train_A['Laptop Compartment'] = train_A['Laptop Compartment'].map({'Yes': 1, 'No': 0})  
train_A['Waterproof'] = train_A['Waterproof'].map({'Yes': 1, 'No': 0})  


from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


target = "Price"
X = train_A.drop(columns = target)
y = train_A[target]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 ,random_state=42)

model = LinearRegression() 
print("Starting model training...")  
model.fit(X_train, y_train)  
print("Model training completed.")


y_pred = model.predict(X_test)  

mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
  
print(f"Mean Absolute Error: {mae}")  
print(f"Mean Squared Error: {mse}")  
print(f"R-squared: {r2}")


test_A = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col="id")  



test_A = optimize_dataframe(test_A)
test_A = fill_nulls(test_A, ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])
test_A['Weight Capacity (kg)'] = test_A['Weight Capacity (kg)'].fillna(value)
test_A['Size'] = test_A['Size'].map(size_mapping) 
test_A = pd.get_dummies(test_A, columns=['Brand', 'Material', 'Style', 'Color'], drop_first=True) 
test_A['Laptop Compartment'] = test_A['Laptop Compartment'].map({'Yes': 1, 'No': 0})  
test_A['Waterproof'] = test_A['Waterproof'].map({'Yes': 1, 'No': 0})  


missing_cols = set(X.columns) - set(test_A.columns)
for col in missing_cols:
    test_A[col] = 0  

test_A = test_A[X.columns]


test_preds = model.predict(test_A)


submission = pd.DataFrame({'id': test_A.index, 'Price': test_preds})
submission.to_csv(r'C:\Users\abhishek\Desktop\submission.csv', index=False)
print("Submission file saved successfully on Desktop.")








