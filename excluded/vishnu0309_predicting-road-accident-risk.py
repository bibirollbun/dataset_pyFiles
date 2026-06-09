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


import os
from pathlib import Path

# Example: Delete all files in the current working directory
files_to_delete = Path(os.getcwd()).glob('*')
for file_path in files_to_delete:
    if file_path.is_file():
        os.remove(file_path)
        print(f"Deleted: {file_path}")
print("Deletion complete.")


#Installing the library 
!pip install catboost


#Importing the library
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Loading the data set 
train_data=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


#First 5 elements of the train data 
train_data.head()


#shape of the train data 
train_data.shape


#Info of the training data 
train_data.info()


#null values in the train and test data 
print(f"Null values in the train data")
print(f"{train_data.isna().sum()}")
print("--------------------------------------")
print(f"Null values in the test data")
test_data.isna().sum()


# Duplicate in the training data  and testing data 
print("Train data duplicate value")
print(f"{train_data.duplicated().sum()}")
print("-"*50)
print("Test data duplicate value")
test_data.duplicated().sum()


#Columns name in the training data 
number_of_columns=train_data.columns
number_of_columns


#Describe the train data 
train_data.describe()


#categorical and continuous columns
categorical_columns=[]
continuous_columns=[]

#Iterating the lopp through the columns
for col in number_of_columns:
    if train_data[col].dtypes=="bool" or train_data[col].dtypes=="object":
     categorical_columns.append(col)
    elif train_data[col].nunique()<5:
     categorical_columns.append(col)
    else:
     continuous_columns.append(col)
print(f"Categorical columns are following:\n{categorical_columns}")
print("-"*100)
print(f"Continous columns are following:\n {continuous_columns}")


#unique value in the train data and test data 
print("Unique values in the categorical columns train data ")
print(f"{test_data[categorical_columns].nunique()}")
print("____________________________________________")
print("Unique values in the categorical columns test data ")
print(f"{train_data[categorical_columns].nunique()}")


#Copy the training data 
df=train_data.copy()
#Dropping the id columns from the df data 
df=df.drop("id",axis=1)


# Plotting the target varibale
import matplotlib.pyplot as plt 
import seaborn as sns 
fig,axes=plt.subplots(1,2,figsize=(18,5))

sns.histplot(df['accident_risk'],ax=axes[0])
axes[0].set_title("Accident Risk distribution")
axes[0].set_xlabel("Accident Risk")

sns.boxplot(df['accident_risk'],ax=axes[1])
axes[1].set_title("Accident Risk Boxplot")
axes[1].set_ylabel("Accident Risk")


continuous_columns=continuous_columns[1:]
fig, axes = plt.subplots(nrows=1, 
                         ncols=len(continuous_columns), 
                         figsize=(15, 5))
for i, col in enumerate(continuous_columns):
    # Use 'x=' to create a horizontal boxplot
    sns.boxplot(x=df[col], ax=axes[i])
    
    # Set a title for each subplot to know which column it is
    axes[i].set_title(f'Distribution of {col}')

# Optional: This cleans up the layout
plt.tight_layout()


# Box plot  for a clear summary
sns.boxplot(x='weather', y='accident_risk', data=df)
plt.title('Accident Risk by Weather')




# Violin plot gives more detail on the distribution
sns.violinplot(x='time_of_day', y='accident_risk', data=df)
plt.title('Accident Risk by Time of Day')


columns_to_plot = [col for col in categorical_columns if col != "num_reported_accidents"]
num_plots = len(columns_to_plot) # This should be 4, based on your error

# 2. Create the figure and axes with the correct size
# You can adjust the figsize for better readability
fig, axes = plt.subplots(nrows=1, 
                         ncols=num_plots, 
                         figsize=(18, 5))

# 3. Use a counter 'i' that only increments when a plot is drawn
i = 0 
for col in categorical_columns:
    if col != "num_reported_accidents":
        # Group, aggregate, and plot onto the current axis (axes[i])
        df.groupby(col).agg({"num_reported_accidents": "count"}).plot(kind="bar", ax=axes[i])
        axes[i].set_title(f'Count by {col}') # Add title for clarity
        i += 1 # Only increment the counter if a plot was successfully added

# Optional: This cleans up the layout
plt.tight_layout()



#Bar plot to check the effect of the speed limit on the accident risk
sns.barplot(x='speed_limit', y='accident_risk', hue='weather', data=df)


#Checking the correlction between the continous variables
sns.heatmap(df[continuous_columns].corr())


df


#initialize the label encoder
le=LabelEncoder()
for col in categorical_columns:
    #Transforming the columns into categorical data 
    df[col]=le.fit_transform(df[col])


#Catboost model defining 
model = CatBoostRegressor(loss_function='RMSE',
                          iterations=100)


#Spliting the dataset between the test data and train data 
x=df.drop("accident_risk",axis=1)
y=df['accident_risk']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#Fitting the model for train and test data
model.fit(x_train,y_train,
          cat_features=categorical_columns
          )


#Prediciting for the test data
y_predict=model.predict(x_test)


#Comaparision df for y_test and prediction column
comparison_df = pd.DataFrame({'y_test': y_test, 'y_predict': y_predict})


comparison_df


#Finding the RMSE for the y_test and y_predict
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")




#initialize the label encoder for the test data 
le=LabelEncoder()
for col in categorical_columns:
    #Transforming the columns into categorical data 
    test_data[col]=le.fit_transform(test_data[col])


test_data


#Copy the test data 
test_data1=test_data.copy()



test_data1.drop("id",axis=1,inplace=True )


#Training on the full train data 
#Catboost model defining 
model_cat = CatBoostRegressor(loss_function='RMSE',
                          iterations=100)
x=df.drop("accident_risk",axis=1)
y=df['accident_risk']
#Fitting the model for train and test data
model_cat.fit(x,y,
          cat_features=categorical_columns
          )



#Predict from the trained model
predictions = model_cat.predict(test_data1)


#Making the data set for the submission
submission=pd.DataFrame({"id":test_data['id'],"accident_risk":predictions})



submission


submission.to_csv('submission.csv',index=False)




