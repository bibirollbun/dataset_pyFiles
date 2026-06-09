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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.stats import skew
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_data.head()


train_data.shape


train_data.info()


test_data.head(2)


test_data.shape


test_data.info()


train_data.describe()


train_data.head(1)


train_data.columns[1:12]


X = train_data.iloc[:,1:12]
y = train_data.iloc[:,-1]


X.head(2)


X.shape


y.head(2)


plt.figure(figsize=(15, 8))

# Create boxplot for all numerical features
sns.boxplot(data=train_data, palette="Set2")

# Customize plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.title("Boxplot to Visualize Outliers", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for readability
plt.show()


import math

# Number of numerical columns
num_cols = len(train_data.columns)

# Calculate number of rows and columns dynamically
rows = math.ceil(num_cols / 3)  # 3 columns per row
cols = min(num_cols, 3)  # Max 3 columns

plt.figure(figsize=(cols * 4, rows * 4))  # Adjust figure size dynamically

# Plot histograms for each numerical column
for i, col in enumerate(train_data.columns, 1):
    plt.subplot(rows, cols, i)  # Dynamically adjust grid size
    sns.histplot(train_data[col], kde=True, bins=30, color="blue")
    
    # Check skewness
    skewness = skew(train_data[col].dropna())  # Drop NaNs to avoid errors
    plt.title(f"{col} (Skew: {skewness:.2f})")

plt.tight_layout()
plt.show()



train_data['rainfall'].value_counts()


train_data.groupby('rainfall').mean()


scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)


standardized_data


# Assuming `X_standardized` is the standardized NumPy array
X_standardized_dataframe = pd.DataFrame(standardized_data, columns=X.columns)

# Display the first few rows
print(X_standardized_dataframe.head())


X = standardized_data


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, stratify = y,random_state = 42)


print(X.shape,X_train.shape,X_test.shape)


classifier = svm.SVC(kernel = "linear")
classifier.fit(X_train, y_train)


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)


print("Accuracy_Score of the training data", training_data_accuracy)


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)


print("Accuracy_Score of the test data", test_data_accuracy)


X_test_data = test_data


X_test_data.head(2)


X_test_data.isnull().sum()


X_test_data = X_test_data.drop(columns = 'id',axis = 1)


from sklearn.impute import SimpleImputer
import pandas as pd

# Step 1: Ensure X_train is a DataFrame before using columns
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)

# Step 2: Handle NaN values while maintaining feature count
imputer = SimpleImputer(strategy="mean")
X_test_data = pd.DataFrame(imputer.fit_transform(X_test_data), columns=X_train.columns)

# Step 3: Ensure test features match training features
if X_test_data.shape[1] > X_train.shape[1]:
    X_test_data = X_test_data.iloc[:, :X_train.shape[1]]  # Remove extra columns

print("Final X_test_data shape:", X_test_data.shape)

# Step 4: Convert to numpy and predict
test_predictions = classifier.predict(X_test_data.to_numpy())



submission_df = pd.DataFrame({"id": test_data.index, "rainfall": test_predictions})
submission_df.to_csv("rainfallprediction1.csv", index=False)
print("Submission file saved as rainfallprediction1.csv")


# Ensure X_test_data and predictions are of the same length
print("Test data index length:", len(test_data.index))
print("Predictions length:", len(test_predictions))

# Create submission file with corrected index
submission_df = pd.DataFrame({
    "id": test_data.index[:len(test_predictions)],  # Ensure same length
    "rainfall": test_predictions
})

# Save CSV file
submission_df.to_csv("rainfallprediction1.csv", index=False)
print("Submission file saved as rainfallprediction1.csv")



submission_data = pd.read_csv("/kaggle/working/rainfallprediction1.csv")
print(submission_data.head())




