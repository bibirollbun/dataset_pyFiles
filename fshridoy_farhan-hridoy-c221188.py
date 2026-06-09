# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Label encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Feature and target split
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'], errors='ignore')
y = train_data['satisfaction']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape features for CNN (samples, features, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# One-hot encode the target labels
y_cat = to_categorical(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Input(shape=(X.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')  # 2 output units for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
print(f"CNN Validation Accuracy: {accuracy_score(y_val_classes, y_pred_classes):.2f}")

# --- Predict on test dataset ---
# Load test dataset
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Apply the same label encoders to test data
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])

# Drop non-feature columns
X_test = test_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Impute missing values and scale
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test = scaler.transform(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict satisfaction
y_test_pred = model.predict(X_test)
y_test_classes = np.argmax(y_test_pred, axis=1)

# Decode predictions back to original satisfaction labels
test_data['satisfaction'] = label_encoders['satisfaction'].inverse_transform(y_test_classes)

# Prepare submission file
submission = test_data[['id', 'satisfaction']]
submission.to_csv("cnn_submission.csv", index=False)

# Display sample predictions
print(submission.head())


submission.to_csv("submission.csv", index=False)



import os
os.listdir()



from google.colab import files
files.download('submission.csv')



from IPython.display import FileLink
FileLink('submission.csv')  # অথবা 'cnn_submission.csv'



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Label encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Feature and target split
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'], errors='ignore')
y = train_data['satisfaction']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape features for CNN (samples, features, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# One-hot encode the target labels
y_cat = to_categorical(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Input(shape=(X.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')  # 2 output units for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
print(f"CNN Validation Accuracy: {accuracy_score(y_val_classes, y_pred_classes):.2f}")

# --- Predict on test dataset ---
# Load test dataset
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Apply the same label encoders to test data
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])

# Drop non-feature columns
X_test = test_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# Impute missing values and scale
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test = scaler.transform(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict satisfaction
y_test_pred = model.predict(X_test)
y_test_classes = np.argmax(y_test_pred, axis=1)

# Decode predictions back to original satisfaction labels
test_data['satisfaction'] = label_encoders['satisfaction'].inverse_transform(y_test_classes)

# Rename 'id' column to 'ID' for submission format
test_data.rename(columns={'id': 'ID'}, inplace=True)

# Prepare submission file
submission = test_data[['ID', 'satisfaction']]
submission.to_csv("cnn_submission.csv", index=False)

# Display sample predictions
print(submission.head())



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")
test_df =  pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Show basic structure
print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape:", test_df.shape)

train_df.head()


print("Train Dataset Info:")
train_df.info()

print("\nTest Dataset Info:")
test_df.info()


print("Missing values in Train Data:\n", train_df.isnull().sum())
print("\nMissing values in Test Data:\n", test_df.isnull().sum())




print("Duplicate rows in Train:", train_df.duplicated().sum())
print("Duplicate rows in Test:", test_df.duplicated().sum())






print("Train Dataset Columns:\n", train_df.columns.tolist())




cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, palette="Set2")
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()




num_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()




for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=train_df, x=col, y='Age', hue='satisfaction')
    plt.title(f'{col} vs Age by Satisfaction')
    plt.xticks(rotation=45)
    plt.show()


# Average age of satisfied vs neutral/dissatisfied customers
train_df.groupby('satisfaction')['Age'].mean()

# Which class has the highest average flight distance
train_df.groupby('Class')['Flight Distance'].mean()

# Average delay by travel type
train_df.groupby('Type of Travel')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean()


# Average flight distance by customer type
train_df.groupby('Customer Type')['Flight Distance'].mean()

# Average inflight service ratings by satisfaction
train_df.groupby('satisfaction')[['Inflight service', 'On-board service', 'Cleanliness']].mean()

# Satisfaction count by gender
train_df.groupby(['Gender', 'satisfaction']).size().unstack()

# Median age by class and travel type
train_df.groupby(['Class', 'Type of Travel'])['Age'].median()


# Top 10 passengers with highest flight distance
train_df.sort_values(by='Flight Distance', ascending=False).head(10)

# Passengers with longest departure delay
train_df.sort_values(by='Departure Delay in Minutes', ascending=False).head(10)


# Passengers who are satisfied and traveled in Business class
train_df[(train_df['satisfaction'] == 'satisfied') & (train_df['Class'] == 'Business')]

# Female passengers under 25 who were neutral or dissatisfied
train_df[(train_df['Gender'] == 'Female') & (train_df['Age'] < 25) & (train_df['satisfaction'] == 'neutral or dissatisfied')]


# Satisfaction across travel type and class
pd.crosstab(train_df['Type of Travel'], train_df['Class'], margins=True)

# Heatmap of satisfaction rate by gender and class
ct = pd.crosstab(train_df['Gender'], train_df['Class'], values=(train_df['satisfaction'] == 'satisfied').astype(int), aggfunc='mean')
sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Satisfaction Rate by Gender and Class")
plt.show()




