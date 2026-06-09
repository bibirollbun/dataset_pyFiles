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

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Read CSV files from Kaggle input directory
train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


print("Train data shape : " , train_data.shape)
print("Test data shape : " , test_data.shape)


print("Train data head : \n" , train_data.head)
print("Test data head : \n" , test_data.head())


print("Train data info: \n" , train_data.info())
print("Test data info: \n" , test_data.info())


print("Train data: \n" , train_data.describe())
print("Test data: \n" , test_data.describe())


# Visualize target class distribution
sns.countplot(x='rainfall', data=train_data, palette='coolwarm')
plt.title("Target Class Distribution - Rainfall")
plt.show()


# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


print("Train data null values sum for each column: \n" , train_data.isnull().sum())
print("Test data null values sum for each column: \n" , test_data.isnull().sum())


test_data['winddirection'].fillna(test_data['winddirection'].mean(), inplace=True)


# Drop unnecessary columns
train_data.drop(columns=['id', 'day'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id', 'day'], inplace=True)


# Separate features and target
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']


from sklearn.feature_selection import SelectKBest, f_classif

# Use SelectKBest to select top 5 best features
k = 8  # Select top 8 features
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)

mask = selector.get_support()        # Get the boolean mask of selected features
selected_features = X.columns[mask]       # Get the selected feature names
print("Selected features:", selected_features)      # Print the selected features


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


outliers = []
for feature in train_data[selected_features].columns:
    Q1 = train_data[feature].quantile(0.25)
    Q3 = train_data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    # Capping and flooring outliers
    train_data[feature] = np.where(train_data[feature] < lower_bound, lower_bound, train_data[feature])
    train_data[feature] = np.where(train_data[feature] > upper_bound, upper_bound, train_data[feature])


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

test_scaled = scaler.transform(test_data[selected_features])


# Base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', C=1, probability=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Meta model (final estimator)
meta_model = LogisticRegression()

# Stacking Classifier
stack_model = StackingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('gb', gb)],
    final_estimator = meta_model
)

# Fit the stacked model
stack_model.fit(X_train, y_train)

# Predict on training data
y_train_pred = stack_model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Classification Report: \n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

# Predict on test data
y_test_pred = stack_model.predict(X_test)
print("\nTesting Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report: \n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


# Predict on Test Data
test_predictions = stack_model.predict(test_scaled)

# Create Submission File
submission = pd.DataFrame({'id': test_ids, 'rainfall': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

