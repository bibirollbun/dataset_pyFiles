# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


Train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
Train


Test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
Test


Train.columns


X_train = Train[['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']]
Y_train = Train['rainfall']
print(X_train[:5])
print(Y_train[:5])


from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
print("X_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)


print(y_train[:20])


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(x_train, y_train)

y_pred = model_rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ðŸ”¹ Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ðŸ”¹ Step 8: Plot Histogram for Predictions
plt.figure(figsize=(6,4))
sns.histplot(y_test, bins=2, discrete=True, color='blue', label='Actual Rainfall', alpha=0.6)
sns.histplot(y_pred, bins=2, discrete=True, color='orange', label='Predicted Rainfall', alpha=0.6)
plt.xticks([0, 1], ["No Rain", "Rain"])
plt.xlabel("Rainfall Occurrence")
plt.ylabel("Count")
plt.title("Actual vs. Predicted Rainfall Occurrence")
plt.legend()


print(y_pred)


X_test = Test[['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']]
print(X_test)


print(" Checking for missing values:\n")
print(X_test.isnull().sum())  # Shows the count of NaN values per column

# ðŸ”¹ Step 3: Replace NaN Values with Column Mean
X_test.fillna(X_test.mean(numeric_only=True), inplace=True)

# ðŸ”¹ Step 4: Verify No Missing Values Left
print("\n Missing values after replacement:\n")
print(X_test.isnull().sum())


output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)


test_probs = model_rf.predict_proba(X_test)[:, 1]  # Extracting probability of rainfall (class 1)

# ðŸ”¹ Create Submission DataFrame
submission = pd.DataFrame({'id': Test['id'], 'rainfall': test_probs})

# ðŸ”¹ Save the file in 'output' directory
submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"\n Submission file saved at: {submission_path}")


submission.head()




