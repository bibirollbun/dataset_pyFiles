!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


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


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv',index_col = 0)
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col=0)


sample_submission.shape


train.shape


test.shape


print(train.isnull().sum())


print(test.isnull().sum())


train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})


test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)


train['Stage_fear'] = train['Stage_fear'].fillna(0.5)
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(0.5)


test['Stage_fear'] = test['Stage_fear'].fillna(0.5)
test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(0.5)


print(train.isnull().sum())


print(test.isnull().sum())


train.info()


train.describe()


test.info()


test.describe()


sns.countplot(data=train, x='Personality', palette='Set2')
plt.title("Class Distribution on the Train Set: Personality Types")
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train_resampled, y_train_resampled)


y_pred = clf.predict(X_val)


print(classification_report(y_val, y_pred))


ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val)


feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Features Influencing Personality Prediction")
plt.show()


X_test = test.copy()
X_test


y_test_preds = clf.predict(X_test)
y_test_preds 


X_test['Personality'] = y_test_preds
X_test['Personality'] = X_test['Personality'].map({0: 'Extrovert', 1: 'Introvert'})
X_test


submission = X_test[['Personality']].copy()
submission.reset_index(inplace=True) 

submission.rename(columns={'id': 'id', 'Personality': 'Personality'}, inplace=True)


submission.to_csv('submission.csv', index=False)

print("Submission file saved successfully!")


from sklearn.metrics import accuracy_score

true_labels = X_test['Personality']
predicted_labels = sample_submission['Personality']

accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


train_data = X_train  
val_data = X_val     

duplicates = val_data.merge(train_data, how='inner')
print(f"Duplicates between train and val: {len(duplicates)}")


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())

