# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train.info()


test.info()


train.shape


train.sample(5)


train['date'] = pd.to_datetime(train['date'])


train['year']=train['date'].dt.year


sales_by_year = train.groupby('year')['num_sold'].sum().reset_index()
sales_by_country = train.groupby('country')['num_sold'].sum().reset_index()
sales_by_store = train.groupby('store')['num_sold'].sum().reset_index()

print("Sales by Year:")
print(sales_by_year.to_string(index=False))


plt.plot(sales_by_year.year,sales_by_year.num_sold, marker='o', linestyle='-', color='b')

# Add labels for each point
for i in range(len(sales_by_year)):
    plt.text(sales_by_year.year.iloc[i], sales_by_year.num_sold.iloc[i], 
             str(sales_by_year.num_sold.iloc[i]), 
             fontsize=9, ha='right', va='bottom')

plt.title('Year vs Sales')
plt.xlabel('Year')
plt.ylabel('Sales')

plt.show()


print("\nSales by Country:")
print(sales_by_country.to_string(index=False))


print("\nSales by Store:")
print(sales_by_store.to_string(index=False))


sales_by_year.head()


# Prepare the data
X = sales_by_year['year'].values.reshape(-1, 1)  # Independent variable (year)
y = sales_by_year['num_sold'].values  # Dependent variable (sales)


# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Get the regression line values
y_pred = model.predict(X)

# Plot the original data and the regression line
plt.scatter(sales_by_year['year'], sales_by_year['num_sold'], color='blue', label='Actual Data')
plt.plot(sales_by_year['year'], y_pred, color='red', label='Regression Line')

# Title and labels
plt.title('Linear Regression: Year vs Sales')
plt.xlabel('Year')
plt.ylabel('Sales')

# Show the legend and plot
plt.legend()
plt.show()

# Output the regression equation and R^2 score
print(f"Linear Regression Equation: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
print(f"R^2 Score: {model.score(X, y):.2f}")



# Predict for the next 4 years
future_years = np.array([sales_by_year['year'].max() + i for i in range(1, 5)]).reshape(-1, 1)
future_sales = model.predict(future_years)

# Output the predictions for the next 4 years
print("\nPredictions for the next 4 years:")
for year, sales in zip(future_years.flatten(), future_sales):
    print(f"Year: {year}, Predicted Sales: {sales:.2f}")


# Plot the original data and the regression line
plt.scatter(sales_by_year['year'], sales_by_year['num_sold'], color='blue', label='Actual Data')
plt.plot(sales_by_year['year'], y_pred, color='red', label='Regression Line')

# Plot future predictions
plt.scatter(future_years, future_sales, color='green', label='Predicted Future Sales')

# Title and labels
plt.title('Linear Regression: Year vs Sales (with Future Predictions)')
plt.xlabel('Year')
plt.ylabel('Sales')

# Show the legend and plot
plt.legend()
plt.show()


# Calculate Mean Absolute Percentage Error (MAPE) for the historical data
mape = np.mean(np.abs((y - y_pred) / y)) * 100


# Output the Mean Absolute Percentage Error (MAPE)
print(f"\nMean Absolute Percentage Error (MAPE) for the training data: {mape:}%")

