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


pip install catboost



df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df


df.isnull().sum()


df.drop(columns=['day'], inplace=True, errors='ignore')  # These may not be useful



df


df['maxtemp'].describe()


df['mintemp'].describe()


X = df.drop(columns=['rainfall'])  # Features
y = df['rainfall']  # Target Variable


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize features



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn import linear_model
# Train Bayesian Ridge Regression
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)

# Get continuous predictions
y_pred = reg.predict(X_test)

# Convert predictions to binary (assuming binary classification)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Compute accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_binary))


import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply LDA
lda = LDA()
lda.fit(X_train, y_train)

# Predict on test data
y_pred = lda.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("LDA Model Accuracy:", accuracy)



from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Initialize the CatBoost classifier
cat = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)

# Fit the model
cat.fit(X_train, y_train)

# Make predictions
y_pred = cat.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df


test_df.fillna({'winddirection': test_df['winddirection'].mean()}, inplace=True)


# Drop unnecessary columns
test_df.drop(columns=['day'], inplace=True, errors='ignore')

# Apply the same scaling as training data
X_test_final = scaler.transform(test_df)  # Use the same scaler from training


X_test_final


# Predict on test data
y_test_pred = rf.predict(X_test_final)  # Ensure X_test is preprocessed like X_train



# Create a DataFrame for submission
submission = pd.DataFrame({'id': test_df['id'], 'rainfall': y_test_pred})

# Save as CSV
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")

