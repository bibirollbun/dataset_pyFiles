'''from google.colab import files
files.upload()  # Upload kaggle.json manually

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json'''


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

%matplotlib inline

# Load and shuffle data
df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").sample(frac=1, random_state=42)
df.head()



print(f"Dataset Shape: {df.shape}")
df.describe()



df.info()


df['is_male'] = df['Sex'].map({'female': 0, 'male': 1})
df.drop('Sex', axis=1, inplace=True)


plt.figure(figsize=(7, 4))
plt.scatter(df['Body_Temp'], df['Calories'], c=df['Age'], cmap='viridis', alpha=0.6)
plt.xlabel("Body Temperature")
plt.ylabel("Calories")
plt.title("Body Temperature vs Calories")
plt.colorbar(label='Age')
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 4))
plt.scatter(df['Duration'], df['Calories'], c=df['Age'], cmap='viridis', alpha=0.6)
plt.xlabel("Duration")
plt.ylabel("Calories")
plt.title("Duration vs Calories")
plt.colorbar(label='Age')
plt.grid(True)
plt.show()



plt.figure(figsize=(7, 4))
plt.scatter(df['Heart_Rate'], df['Calories'], c=df['Age'], cmap='viridis', alpha=0.6)
plt.xlabel("Heart_Rate")
plt.ylabel("Calories")
plt.title("Heart_Rate vs Calories")
plt.colorbar(label='Age')
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Defining input features and target variable
features = ['Age','Height','Weight','Duration', 'Heart_Rate', 'Body_Temp','is_male']
target = 'Calories'

# Creating a copy of the input features
inputs = df[features].copy()

# Copying target values
outputs = df[target].copy()

# Normalizing the input features using Min-Max scaling
inputs = preprocessing.MinMaxScaler().fit_transform(inputs)

# Splitting data into training and validation sets (first 700,000 for training, the rest for validation)
X_train, X_val = inputs[:700000].copy(), inputs[700000:].copy()
y_train, y_val = outputs[:700000].copy(), outputs[700000:].copy()

# Creating and training the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=14, max_features=5, random_state=42)
rf.fit(X_train, y_train)

# Predicting on the validation set
preds_rf = rf.predict(X_val)

# Calculating and printing the Mean Absolute Error (MAE)
print(f"Random Forest MAE: {mean_absolute_error(y_val, preds_rf):.4f}")
# Calculating and printing the Mean Squared Log Error (RMSLE)
print(f"Random Forest RMSLE: {np.sqrt(mean_squared_log_error(y_val, preds_rf)):.4f}")


show = pd.DataFrame({
    'Actual': y_val,
    'Predicted':preds_rf,
})
verify = y_val.reset_index(drop=True).copy()
verify['Actual'] = show['Actual'].reset_index(drop=True)
verify['Predicted'] = show['Predicted'].reset_index(drop=True)
verify['Error'] = verify['Actual'].copy() - verify['Predicted'].copy()
show['Error'] = show['Actual'] - show['Predicted']


g = plt.hist(show["Error"], bins=100)

