# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data.head()


test_data.head()


#import seaborn as sns
#import matplotlib.pyplot as plt

# Створюємо графік 
#sns.countplot(x='maxtemp', hue='humidity', data=train_data)
#plt.title('humidity depends on maxtempereture')
#plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Кореляційна матриця 
corr_matrix = train_data[['maxtemp', 'humidity', 'rainfall', 'sunshine', 'mintemp', 'windspeed', 'temparature', 'dewpoint']].corr()

# Візуалізуємо кореляційну матрицю
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Кореляційна матриця для maxtemp, humidity, rainfall, sunshine та mintemp")
plt.show()






X = train_data.drop(columns=['rainfall', 'id', 'day'])  
y = train_data['rainfall'] 


from sklearn.model_selection import train_test_split

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Видаляємо ті ж самі колонки з test_data
test_features = test_data.drop(columns=['id', 'day'])


# Check the size
print(X_train.shape, X_test.shape)


# from sklearn.linear_model import LinearRegression

# Create the model
# model = LinearRegression()

# Train the model
# model.fit(X_train, y_train)

# Predict on test data
# y_pred = model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#test_predictions = model.predict_proba(test_features)[:, 1]  # Нам треба ймовірність для класу 1



from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#submission = pd.DataFrame({
#    "id": test_data["id"],
#    "rainfall": test_predictions
#})
#submission.to_csv("submission.csv", index=False)
#print("Submission file saved successfully!")


#from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate error metrics
##mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)

#print(f"Mean Absolute Error (MAE): {mae}")
#print(f"Mean Squared Error (MSE): {mse}")


#print("Train feature columns:", X_train.columns.tolist())
#print("Test feature columns:", test_features.columns.tolist())


#test_features = test_features[X_train.columns]


print(test_features.dtypes)
print(X_train.dtypes)


print(test_features.isnull().sum())


test_features = test_features.fillna(test_features.mean())


#baseline_pred = np.full_like(y_test, y_train.mean())
#baseline_mae = mean_absolute_error(y_test, baseline_pred)
#print(f"Baseline MAE: {baseline_mae}")


##from sklearn.metrics import r2_score
#r2 = r2_score(y_test, y_pred)
#print(f"R² Score: {r2}")


#import matplotlib.pyplot as plt

#plt.scatter(y_test, y_pred, alpha=0.5)
#plt.xlabel("Actual Values")
#plt.ylabel("Predicted Values")
#plt.title("Actual vs. Predicted")
#plt.show()


# from sklearn.ensemble import RandomForestRegressor

# rf_model = RandomForestRegressor()
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)

# print("MAE:", mean_absolute_error(y_test, y_pred_rf))
# print("MSE:", mean_squared_error(y_test, y_pred_rf))


# print(f"test_data shape: {test_data.shape}")  # Should match number of rows in test.csv
# print(f"test_predictions shape: {test_predictions.shape}") 


# Drop the "id" column before making predictions
# test_features = test_data.drop(columns=["id"]) 

# Predict on the entire test dataset
# test_predictions = model.predict(test_features)

# Ensure predictions are in the right shape
# test_predictions = test_predictions.flatten()

# Check the shape again
# print(f"Fixed test_predictions shape: {test_predictions.shape}")


# Ensure model is trained before making predictions
#test_predictions = model.predict(X_test)  # Use trained model

# Ensure predictions are within a valid range if necessary
#test_predictions = test_predictions.clip(0, 1)

# Create submission DataFrame
# submission = pd.DataFrame({
#     "id": test_data["id"],  
#     "rainfall": test_predictions
# })

# Save the submission file
# submission.to_csv("submission.csv", index=False)

# print("Submission file saved successfully!")


# Прогноз на змаганні тестових даних
# test_predictions = model.predict(test_features)


# 1. Робимо передбачення для тестового набору Kaggle
test_predictions = model.predict(test_features)

# 2. Формуємо submission DataFrame
submission = pd.DataFrame({
    "id": test_data["id"],  # ID з тестового набору
    "rainfall": test_predictions  # Передбачені значення
})

# 3. Зберігаємо у CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")

