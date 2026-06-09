import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

import warnings
warnings.filterwarnings('ignore')


# load data
prediction = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# copy my data
prediction_copy = prediction.copy()

test_copy = test_data.copy()

prediction_copy.head()


# inspection of data
prediction_copy.info()
prediction_copy.columns
prediction_copy['weather'].value_counts()
prediction_copy['road_type'].value_counts()
prediction_copy.isnull().sum()
prediction_copy.duplicated().sum()


# data cleaning and transformation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
prediction_copy['weather'] = le.fit_transform(prediction_copy['weather'])
prediction_copy['road_type'] = le.fit_transform(prediction_copy['road_type'])
prediction_copy['time_of_day'] = le.fit_transform(prediction_copy['time_of_day'])
prediction_copy['lighting'] = le.fit_transform(prediction_copy['lighting'])
prediction_copy['road_signs_present']= le.fit_transform(prediction_copy['road_signs_present'])
prediction_copy['school_season']= le.fit_transform(prediction_copy['school_season'])
prediction_copy['holiday'] = le.fit_transform(prediction_copy['holiday'])
prediction_copy['public_road'] = le.fit_transform(prediction_copy['public_road'])
prediction_copy
prediction_copy.shape


# lets look at the correlation of this data
prediction_copy.corr()


# get our array

array = prediction_copy.values
array.shape
X = array[:,0:13]
Y = array[:,13]


test_copy.head()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_copy['weather'] = le.fit_transform(test_copy['weather'])
test_copy['road_type'] = le.fit_transform(test_copy['road_type'])
test_copy['time_of_day'] = le.fit_transform(test_copy['time_of_day'])
test_copy['lighting'] = le.fit_transform(test_copy['lighting'])
test_copy['road_signs_present']= le.fit_transform(test_copy['road_signs_present'])
test_copy['school_season']= le.fit_transform(test_copy['school_season'])
test_copy['holiday'] = le.fit_transform(test_copy['holiday'])
test_copy['public_road'] = le.fit_transform(test_copy['public_road'])
test_copy.shape


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3333339771, random_state=42)


# training sets
print(X_train.shape, Y_train.shape)

# testing sets
print(X_test.shape, Y_test.shape)

print(len(test_copy))


# c) Import Scientific Algorithms
# This Algorithms will be used to create a model that will learn from the Training Data
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
# Linear Regression
from sklearn.linear_model import LinearRegression
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor

# d) Cross - Validation(Hyperparameter Tuning)
# This process validates and picks the right Algorithm that will produce a higher Accuracy.
models = []
models.append(('D Trees', DecisionTreeRegressor()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('LR', LinearRegression()))


# Import Cross Validation and KFOLD
from sklearn.model_selection import cross_val_score, KFold
# Create a for loop so that each model is tested in turn
for name, model in models:
          # Here we do a 10 split K-FOLD
          kfold = KFold(n_splits = 10, random_state=42, shuffle=True)

          # We get the results for each Fold, Train and Test using the Folds
          cv_results = cross_val_score(model, X_train, Y_train, cv = kfold,
          scoring='r2')

          # Get the average of all Folds for current model in the loop
          print(name, ' Results:= ', cv_results.mean())




# selct our model to use

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)


# model Evaluation
models_prediction = model.predict(X_test)
models_prediction

print("\n\n")

print("The Correct Answer is : ", Y_test)

print("\n\n")

# Therefore, we will a scientific metric called r2_score, which is the distance between the model_prediction and Y_test result
from sklearn.metrics import r2_score
print("The Accuracy of the Model is: ", r2_score(Y_test, models_prediction) * 100,  "%")



len(test_copy)

print("Test data rows:", len(test_copy))
print("Predictions rows:", len(models_prediction))
output = pd.DataFrame( {'id':test_copy.id, 'Accident':models_prediction } )
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


