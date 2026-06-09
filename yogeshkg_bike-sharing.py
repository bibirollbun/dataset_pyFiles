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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 2. Load the dataset
data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')


# 3. Display basic info (optional for understanding)
print(data.info())


# 4. Extract features for a basic regression model
# We'll convert 'datetime' to ordinal (for numeric input to model) but not split it
data['datetime_ordinal'] = pd.to_datetime(data['datetime']).map(pd.Timestamp.toordinal)


X = data[['datetime_ordinal']]
y = data['count']


# 6. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 7. Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Predict and evaluate the model
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f'RMSE: {rmse:.4f}')


# 9. Plot count vs datetime
plt.figure(figsize=(16, 8))
plt.plot(pd.to_datetime(data['datetime']), data['count'], color='blue', label='Bike Count')
plt.xlabel('Datetime')
plt.ylabel('Count')
plt.title('Bike Sharing Demand Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the external test dataset
test_data = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

# Convert 'datetime' to ordinal format (same as training preprocessing)
test_data['datetime_ordinal'] = pd.to_datetime(test_data['datetime']).map(pd.Timestamp.toordinal)

# Use the trained model to predict counts
X_final = test_data[['datetime_ordinal']]
y_final_pred = model.predict(X_final)

# Create a submission DataFrame
submission = pd.DataFrame({
    'datetime': test_data['datetime'],
    'count': y_final_pred
})

# Ensure no negative predictions (optional but recommended for this dataset)
submission['count'] = submission['count'].apply(lambda x: max(0, x))

# Save the predictions to a CSV file for Kaggle submission
submission.to_csv('bike_demand_submission.csv', index=False)
print("Submission file created: submission.csv")


