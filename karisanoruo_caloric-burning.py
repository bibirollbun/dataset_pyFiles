#Imports

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


#Load Data (with saved file)

# train = pd.read_csv("train.csv",index_col='id') 
# test = pd.read_csv("test.csv",index_col='id') 


#Load Data (With API)
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')


#First Six Rows
print(train.head())


#First Six Rows of Test
print(test.head())


#Find out how many rows
train.shape


test.shape


# Check for missing values

missing = train.isnull().sum()

print("Missing values in each column:")
print (missing)


test.isna().sum()


# Remove duplicate rows
train = train.drop_duplicates( )
print("DataFrame after removing duplicates: ")
print(train.info())


train.describe()


test.describe()


train_new = train.select_dtypes(exclude='object') #Removes Categorical Columns (in this case, 'Sex')

#for loop that plots a new histogram for each column left
for column in train_new:
    fig, ax = plt.subplots(figsize=(18, 5))
    fig = sns.histplot(data=train_new, x=column, bins=50, kde=True)
    plt.show()


# Distribution Plot

sns.displot(train, x="Calories", hue="Sex", multiple="dodge")

sns.displot(train, x="Calories", hue="Sex", kind="kde")



cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
fig, ax = plt.subplots(4,2,figsize=(8,18))
ax = ax.flatten()
for i,col in enumerate(cols):
    sns.kdeplot(data=train,x=col,ax=ax[i])
    sns.kdeplot(data=test,x=col,color='r',ax=ax[i])
    ax[i].set_yticks([])
    ax[i].set_title(col)

sns.kdeplot(data=train,x='Calories',ax=ax[-1])
ax[-1].set_yticks([])
ax[-1].set_title('Calories')

plt.suptitle('Distributions')
plt.tight_layout()
plt.show()


plt.figure(figsize = (30,20))
sns.heatmap(train.corr(numeric_only = True), annot = True, cmap = 'Reds')
plt.show


sns.violinplot(x=train["Age"], inner="quart")


sns.violinplot(x=train["Body_Temp"], inner="quart")


sns.violinplot(x=train["Duration"], inner="quart")


sns.violinplot(x=train["Heart_Rate"], inner="quart")


train_bin = pd.get_dummies(train, columns=['Sex'], drop_first=True, dtype=int) #Removes the Sex Column while adding a One Hot Encoded sex column
print(train_bin.head())


y = train['Calories']
X = train_bin.drop('Calories', axis=1)


# Creating a column BMI
X["BMI"] = X["Weight"]/(X["Height"]/100)**2

#Create a Column Intensity
X["Intensity"] = X["Duration"] * X["Heart_Rate"]

#Create a Column Weight-Based Intensity
X["Weight-Based Intensity"] = X["Duration"] * X["Heart_Rate"] * X["Weight"]



# def BMR_male(weight, height, age):
#     BMR_m = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
#     return BMR_m

# def BMR_female(weight, height, age):
#     BMR_f = 447.593 + (9.247 * weight) + (3.098 * height) - (4.33 * age)
#     return BMR_f

# X["Metabolic Rate"] = [BMR_male(X["Weight"], X["Height"], X["Age"]) if sm == 1 else BMR_female(X["Weight"], X["Height"], X["Age"]) for sm in X["Sex_male"]]


#FE for Test


#Load Test and do the same binning to make things smoother
test_bin = pd.get_dummies(test, columns=['Sex'], drop_first=True, dtype=int)

# Creating a column BMI
test_bin["BMI"] = test_bin["Weight"]/(test_bin["Height"]/100)**2

#Create a Column Intensity
test_bin["Intensity"] = test_bin["Duration"] * test_bin["Heart_Rate"]

#Create a Column Weight-Based Intensity
test_bin["Weight-Based Intensity"] = test_bin["Duration"] * test_bin["Heart_Rate"] * test_bin["Weight"]


#Print
X.head()



test_bin.head()


y.head()


#Model Import

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
model = DecisionTreeRegressor(max_leaf_nodes = 5000)
# Fit model
model.fit(train_X, train_y)


#Model Accuracy

predicted_calories = model.predict(X)
mean_absolute_error(y, predicted_calories)

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


#Cross Validation

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)


from sklearn.model_selection import cross_val_score 
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 

print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")




test['Calories'] = model.predict(test_bin)
test['Calories'].to_csv(f'FinalCalorieSubmission.csv')

