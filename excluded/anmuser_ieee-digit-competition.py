import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/ieee-digit-competition-2/train.csv')
test = pd.read_csv('/kaggle/input/ieee-digit-competition-2/test.csv')
sam = pd.read_csv('/kaggle/input/ieee-digit-competition-2/sample_submission.csv')


train.head(3)


train.isna().sum()


X = train.drop(columns=['label'])
y = train['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy: .2f} %')


print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Divorce Status Classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.grid(False)
plt.show()


Predictions = rf_model.predict(test)


sam['Label']= Predictions
sam.to_csv('submission.csv',index=False)


sam.head()

