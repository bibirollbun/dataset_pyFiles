import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import normalize, scale


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



train_df=pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv')


train_df.head()


train_df=train_df.drop('id', axis=1)


train_df.info()


train_df.shape


train_df.describe()


train_df['Sex'].value_counts()


train_df['Age'].value_counts()


train_df.corr(numeric_only=True)


train_df.isnull().sum()


# Histogram of Listening Time
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# Histogram of Sex
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Sex'], bins=20)
plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# Boxplot of Age by Length
plt.figure(figsize=(12, 6))
sns.boxplot(x='Length', y='Age', data=train_df)
plt.title('Age by Length')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# Boxplot of Age by Diameter
plt.figure(figsize=(12, 6))
sns.boxplot(x='Diameter', y='Age', data=train_df)
plt.title('Age by Diameter')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# Label Encoding
train_df['Sex'] = train_df['Sex'].map({'I': 0, 'F': 1, 'M': 2})


# Set style
sns.set(style="whitegrid")

# Plot histograms for numerical features
train_df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()




Q1 = train_df.quantile(0.25)
Q3 = train_df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outlier_condition = ((train_df < (Q1 - 1.5 * IQR)) | (train_df > (Q3 + 1.5 * IQR)))
outliers = outlier_condition.any(axis=1)


train_df = train_df[~outliers]


# Splitting the dataset
X = train_df.drop('Age', axis=1)
y = train_df['Age']


scaler = StandardScaler()

# Fit and transform the data
Scaled_X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(Scaled_X, y, test_size=0.2, random_state=42)


# Training the model
model_RF = RandomForestRegressor(random_state=42)


# Fit the model
model_RF.fit(X_train, y_train)


# Evaluating the model
y_pred = model_RF.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")


model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(24, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)  # Output layer
])


# Compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])


# Train
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

print(f'R² Score: {r2}')




