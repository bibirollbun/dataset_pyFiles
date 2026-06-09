# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
path_set=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path_set.append(str(os.path.join(dirname, filename)))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

%matplotlib inline 
import seaborn as sns
sns.set()
path_set


sample_sub=pd.read_csv(path_set[0])
train=pd.read_csv(path_set[1])
test=pd.read_csv(path_set[2])
train_extra=pd.read_csv(path_set[3])


train=pd.concat([train,train_extra],axis=0)
train=train.reset_index(drop=True)


train.head()


train.shape


sns.scatterplot(y="Weight Capacity (kg)",x="Price",hue="Laptop Compartment",data=train[train["Style"]=="Tote"][0:3000])


sns.lmplot(y="Compartments",x="Price",hue="Size",data=train[train["Style"]=="Tote"][0:300])



plt.figure(figsize=(10,10))
sns.swarmplot(x="Brand",y="Price",hue="Waterproof",data=train[33000:36000])



sns.histplot(x="Price",hue="Style",data=train[0:3000])



sns.kdeplot(train["Compartments"],shade=True)


sns.distplot(train["Price"])


sns.jointplot(x=train["Weight Capacity (kg)"][0:300],y=train["Price"][0:300],kind="kde")


train["dif"]=train["Weight Capacity (kg)"]*train["Compartments"]


train.isna().sum()


train["Weight Capacity (kg)"].corr(train["Price"])


train.info()


for col in train.columns:
    if(train[col].dtypes=="float64"):
        train[col]=train[col].fillna(train[col].mean())
        #print(train[col].mode())
    else:
        train[col]=train[col].fillna(train[col].mode()[0])


train.info()


categorical_columns = train.select_dtypes(include=['object']).columns
cat_cols=categorical_columns.tolist()
print(cat_cols)


target=train["Price"]


train_d=train.drop(["id","Price"],axis=1)


train_d





tn=pd.get_dummies(train_d,columns=cat_cols,dtype=int,drop_first=True)


sns.histplot(x="Brand",hue="Style",data=train,palette=["red","green","blue"])


col="Style"

df = train

# Get the top 5 most frequent categories
top_5_categories = df[col].value_counts().index

# Filter the DataFrame to include only the top 5 categories
df_top_5 = df[df[col].isin(top_5_categories)]

# Calculate the mean value for each category
mean_values = df_top_5.groupby(col)["Price"].mean()

# Plotting
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=mean_values.index, y=mean_values.values, palette='viridis')
plt.title('Bar Plot of Top 5 Most Frequent Categories with Mean Values')
plt.xlabel('Category')
plt.ylabel('Mean Value')

# Annotate the mean values inside the bars
for i, mean_value in enumerate(mean_values):
    ax.text(i, mean_value / 2, f'{mean_value:.2f}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Example dataset

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tn, target, test_size=0.2, random_state=42)

# Initialize the XGBRegressor
model = XGBRegressor(
    n_estimators=25,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Root of Mean Squared Error: {sqrt(mse)}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example dataset

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tn, target, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=50,  # Number of trees
    max_depth=10,      # Maximum depth of each tree
    min_samples_split=2,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f" Root of Mean Squared Error: {sqrt(mse)}")




