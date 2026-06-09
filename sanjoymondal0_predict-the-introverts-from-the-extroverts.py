import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sample_data = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


print(sample_data.head())
print(test_data.head())
print(train_data.head())


sample_data.head()


test_data.isnull().sum()


train_data.shape


train_data.isnull().sum()


train_data.info()


# Map 'Drained_after_socializing' to binary: Yes=1, No=0 (or customize based on your data)
train_data['Drained_after_socializing'] = train_data['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
test_data['Drained_after_socializing'] = test_data['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Now fill missing values with median
columns_to_fill = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

for col in columns_to_fill:
    median_train = train_data[col].median()
    train_data[col].fillna(median_train, inplace=True)
    test_data[col].fillna(median_train, inplace=True)



print(train_data['Stage_fear'].isnull().sum())
print(train_data['Stage_fear'].unique())
train_data['Stage_fear'] = train_data['Stage_fear'].map({'Yes': 1, 'No': 0})
test_data['Stage_fear'] = test_data['Stage_fear'].map({'Yes': 1, 'No': 0})
train_data['Stage_fear'].fillna(train_data['Stage_fear'].mode()[0], inplace=True)
test_data['Stage_fear'].fillna(test_data['Stage_fear'].mode()[0], inplace=True)


print(train_data[columns_to_fill].isnull().sum())
print(test_data[columns_to_fill].isnull().sum())



train_data.head()





from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['Personality'] = le.fit_transform(train_data['Personality'])


train_data['Personality']


plt.figure(figsize=(10,8))
sns.countplot(x = 'Personality',data =train_data,palette = 'viridis')
plt.xlabel("Personality",fontweight = 'bold',size=16)
plt.show()


# separate train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop('Personality',axis=1),
    train_data['Personality'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


X_train.isnull().sum()


y_train.isnull().sum()


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc1}")


X_real_test = test_data
X_real_test = sc.transform(X_real_test)
y_real_pred = classifier.predict(X_real_test)

# Step 5: Save or view predictions
print(y_real_pred)


# sample_data['Personality'] = y_real_pred.astype(int)
# sample_data.to_csv('submission.csv', index=False)
# from IPython.display import FileLink
# FileLink('submission.csv')

import pandas as pd

# Make sure 'test' has 'id' and 'y_real_pred' is your prediction array
submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': y_real_pred
})

# Fix data type if needed
submission['Personality'] = submission['Personality'].astype(int)

# Check shape
assert submission.shape[0] == 6175, f"Expected 6175 rows, got {submission.shape[0]}"

# No NaNs
assert submission.isnull().sum().sum() == 0, "There are missing values!"

# Save without index
submission.to_csv('submission.csv', index=False)

from IPython.display import FileLink
FileLink('submission.csv')


## SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.show()
acc2 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc2}")


## K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.show()
acc3 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc3}")


## Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.show()
acc4 = accuracy_score(y_test, y_pred)
print(f"Accuracy score : {acc4}")


## Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc5 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc5}")


## Step 1: Evaluate with More Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


## Step 2: Error Analysis
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()




