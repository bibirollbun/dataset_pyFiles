import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
clf = DecisionTreeClassifier()


train_data=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv",header=None)
test_data=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv",header=None)
train_labels=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv",header=None)


train_labels.head()


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)


clf.fit(X_train,y_train)
rf.fit(X_train,y_train)


y_pred1= clf.predict(X_test)
y_pred2=rf.predict(X_test)
print(f'Accuracy Decision Tree: {accuracy_score(y_test, y_pred1)}')
print(f'Accuracy Random Forest: {accuracy_score(y_test, y_pred2)}')


train_data


train_labels


test_data


for col in X_train.columns:
    if col not in test_data.columns:
        test_data[col] = np.nan 
y_test_pred = rf.predict(test_data)


 range(len(test_data))


submission = pd.DataFrame({
    'id': range(9000), # Replace with the appropriate ID column name from the test set
    'target': y_test_pred   # Predictions
})


submission.to_csv('submission.csv', index=False)




