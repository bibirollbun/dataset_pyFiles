# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


url = "/kaggle/input/forest-cover-type-prediction/train.csv"
df = pd.read_csv(url)
df.head()


X = df.drop(columns = ['Id' , 'Cover_Type'])
Y = df['Cover_Type']


X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size =0.2 , random_state =42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train decision tree classifier
dtc = DecisionTreeClassifier(max_depth=10 , random_state =42)
dtc.fit(X_train_scaled, Y_train)
# make predictions
dtc_predictions = dtc.predict(X_test_scaled)
dtc_accuracy = accuracy_score(Y_test, dtc_predictions)
print("Decision Tree Classifier Accuracy:", dtc_accuracy)
print("Classification Report:\n", classification_report(Y_test, dtc_predictions))


#using KNN classsifier
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
knn_predictions = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(Y_test, knn_predictions)
print("KNN Classifier Accuracy:", knn_accuracy)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
print("ðŸ“Š Decision Tree Classification Report:")
print(classification_report(Y_test, dtc_predictions))

cm_dt = confusion_matrix(Y_test, dtc_predictions)
ConfusionMatrixDisplay(confusion_matrix=cm_dt).plot()
plt.title("Decision Tree - Confusion Matrix")
plt.show()


print("ðŸ“Š KNN Classification Report:")
print(classification_report(Y_test, knn_predictions))

cm_knn = confusion_matrix(Y_test, knn_predictions)
ConfusionMatrixDisplay(confusion_matrix=cm_knn).plot()
plt.title("KNN - Confusion Matrix")
plt.show()

