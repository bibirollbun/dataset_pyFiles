# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

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


train = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/train (1).csv')
test = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/test (1).csv')
data_dict = pd.read_excel('/kaggle/input/car-price-prediction-khazani-ahmed/Data_dict.xlsx')
sample = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/sample_submission (2).csv')


train.isnull().mean() * 100
test.isnull().mean() * 100


train['cylindernumber'].unique()
train['cylindernumber'] = train['cylindernumber'].str.strip()
test['cylindernumber'] = test['cylindernumber'].str.strip()


#removing car name as we have no guarantee that those will be in the new dataset
train_encoded = pd.get_dummies(train, columns=['fueltype','aspiration','doornumber','drivewheel'])
test_encoded = pd.get_dummies(test, columns=['fueltype','aspiration','doornumber','drivewheel'])


#annoyingly, it seems that in cylindernumber we have these values typed out vs. anything else. Let me play with that a little bit. If we look up, we can see the unique cylinder values

num_word_mapping = {
    'two': 2,
    'three':3,
    'four':4,
    'five': 5,
    'six': 6,
    'eight':8,
    'twelve':12
}

train_encoded['cylindernumber'] = train_encoded['cylindernumber'].map(num_word_mapping)
test_encoded['cylindernumber'] = test_encoded['cylindernumber'].map(num_word_mapping)

train_encoded = train_encoded.drop(columns = ['CarName', 'carbody', 'enginelocation', 'enginetype', 'fuelsystem'])
test_encoded = test_encoded.drop(columns = ['CarName', 'carbody', 'enginelocation', 'enginetype', 'fuelsystem'])


from sklearn.linear_model import LinearRegression #this loads in the structure linear regression model itself. It is an ordinary least squares model, which just means that by default this linear regression function will attempt to fit each column into a straight line on the axis and minimize the error. It will iterate through options until it finds the one with the least amount of error.
from sklearn.model_selection import train_test_split #train_test_split gives us the most common way to split a dataset. You don't need this, but it makes it easier.

trainX = train_encoded.drop(columns = ['price']) #cannot have our predicted value in the training data for X
trainY = train_encoded['price'] # putting the predicted value in the y (outcome) variable

X_train, X_val, y_train, y_val = train_test_split(trainX, trainY, test_size = .2, random_state = 47) # the random state is the seed to split these datasets on so that this is a repeatable process. The test size is 20%, which is pretty normal here.


model = LinearRegression() #have to initialize the function on the model before running
model.fit(trainX, trainY) # fitting the model

score = model.score(X_val, y_val) 
print(f'Model R^2 Score: {score:.2f}') # this score is the R-squared score. R-squared is a measure of fit commonly used in basic statistics. It tells us the proportion of variance that is accounted for by our x validation set.


import matplotlib.pyplot as plt
# Sort data by car_ID for a clean plot
sorted_indices = np.argsort(trainX["car_ID"].values)  # Sort indices based on car_ID
X_sorted = trainX.iloc[sorted_indices]  # Sort trainX rows based on car_ID
y_sorted = trainY.iloc[sorted_indices]  # Sort trainY to match
y_pred_sorted = model.predict(X_sorted)  # Get predictions for sorted X

# Plot actual prices
plt.scatter(X_sorted["car_ID"], y_sorted, color='blue', label='Actual Prices', alpha=0.6)

# Plot predicted prices as a single line
plt.plot(X_sorted["car_ID"], y_pred_sorted, color='red', linewidth=2, label='Predicted Prices')

# Labels and title
plt.xlabel("Car ID")
plt.ylabel("Price")
plt.title("Predicted vs Actual Car Prices")
plt.legend()
plt.show()



y_pred_train = model.predict(trainX)
y_pred = model.predict(test_encoded)


from sklearn.metrics import mean_squared_error
#playing around with AIC and BIC
num_params = len(model.coef_)+1
print('Number of parameters: %d' % (num_params))

#calculate error
mse = mean_squared_error(trainY, y_pred_train)
mse


from math import log
# calculate AIC for regression
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) +2 * num_params
    return aic
aic = calculate_aic(len(trainY), mse, num_params)
print('AIC: %.3f' % aic)


#calculate BIC
def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic

bic = calculate_bic(len(trainY), mse, num_params)
print('BIC: %.3f' % bic)


submission = pd.DataFrame({
    'car_ID': test_encoded['car_ID'],
    'PredictedPrice': y_pred
})

submission.to_csv('submission.csv', index=False)






