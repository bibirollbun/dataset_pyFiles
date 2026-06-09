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
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import r2_score ,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


X = train.drop(columns=['Listening_Time_minutes', 'id', 'Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time', 'Genre', 'Episode_Sentiment'])
Y = train['Listening_Time_minutes']


# Combine X and y to drop rows with any missing value
df = pd.concat([X, Y], axis=1)
df = df.dropna()

# Separate again
X = df.drop(columns=['Listening_Time_minutes'])
Y = df['Listening_Time_minutes']
from sklearn.impute import SimpleImputer

# Impute numerical features with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)





# Assuming your original dataset is called 'data' and you're predicting 'Listening_Time_minutes'
X = df.drop(['Listening_Time_minutes'], axis=1)
Y = df['Listening_Time_minutes']

# Combine and drop missing values
df = pd.concat([X, Y], axis=1)
df = df.dropna()

# Separate again
X = df.drop(['Listening_Time_minutes'], axis=1)
Y= df['Listening_Time_minutes']

# Encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# Train the model
model = LinearRegression()
model.fit(X_encoded, Y)

# Predict and evaluate
y_pred = model.predict(X_encoded)
mse = mean_squared_error(Y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")



train.info()



imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Now split and train
X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
mse=mean_squared_error(Y_pred,Y_test)
r2=r2_score(Y_pred,Y_test)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Coefficients
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.3f}")


# Combine train and test
combined = pd.concat([X, test], axis=0)

# Fill missing numerical values with column mean
combined = combined.fillna(combined.mean(numeric_only=True))

# Encode categorical variables
combined_encoded = pd.get_dummies(combined, drop_first=True)

# Split back into train and test
X_encoded = combined_encoded.iloc[:len(X), :]
test_encoded = combined_encoded.iloc[len(X):, :]

# Fit the model
model = LinearRegression()
model.fit(X_encoded, Y)

# Predict
test_preds = model.predict(test_encoded)

# Prepare submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)




Y_pred = model.predict(X_encoded.iloc[X_train.shape[0]:, :])  # adjust if you have a separate validation set

# 1. Scatter plot: Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=Y, y=model.predict(X_encoded))
plt.xlabel("Actual Listening_Time_minutes")
plt.ylabel("Predicted Listening_Time_minutes")
plt.title("Actual vs Predicted Listening Time")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')  # diagonal line
plt.show()

# 2. Residual plot: Errors distribution
residuals = Y - model.predict(X_encoded)
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=50)
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.show()


submission




