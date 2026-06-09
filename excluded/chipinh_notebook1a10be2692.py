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


test_file = pd.read_csv('/kaggle/input/csp-iris-dataset/Iris_test.csv')
print(test_file)
print('\n')
train_file = pd.read_csv('/kaggle/input/csp-iris-dataset/Iris_train.csv')
print(train_file)
sub_file = pd.read_csv('/kaggle/input/csp-iris-dataset/Iris_sample_submission.csv')
print(sub_file)
print(train_file.columns)



# Features and labels
X_train = train_file.drop('Species', axis=1).values
y_train = train_file['Species'].values
X_test = test_file[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# Import KNN
from sklearn.neighbors import KNeighborsClassifier

# Create and train the model
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
y_pred = (knn_model.predict(X_test))
submission = pd.DataFrame({
    "Id": test_file["Id"],   # take the Id column from test CSV
    "Species": y_pred        # predicted labels
})
# Optional: save to CSV
submission.to_csv("submission.csv", index=False)
print(submission)


