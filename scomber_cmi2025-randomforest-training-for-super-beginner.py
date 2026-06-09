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


###importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


###Importing the dataset
train0 = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')

train = train0.drop(["row_id","sequence_id","behavior","phase","gesture"], axis=1)


### Splitting features and labels
train_x_all = train.drop(columns=["sequence_type"])
train_y = LabelEncoder().fit_transform(train["sequence_type"])

###Endoding 'subject' and 'orientation'
for col in ["subject", "orientation"]:
    train_x_all[col] = LabelEncoder().fit_transform(train_x_all[col])

###splittind dataset for training and validation
##Stratifying the dataset by 'subject'
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_idx, valid_idx in gss.split(train_x_all, train_y, groups=train["subject"]):
    x_train, x_valid = train_x_all.iloc[train_idx], train_x_all.iloc[valid_idx]
    y_train, y_valid = train_y[train_idx], train_y[valid_idx]


# Deleting Nan
y_train = pd.Series(y_train, index=x_train.index)
y_valid = pd.Series(y_valid, index=x_valid.index)

x_train_dropna = x_train.dropna()
y_train_dropna = y_train.loc[x_train_dropna.index]

x_valid_dropna = x_valid.dropna()
y_valid_dropna = y_valid.loc[x_valid_dropna.index]


###Training a RandomForest model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

clf.fit(x_train_dropna, y_train_dropna)

###Prediction with the trained RandomForest model
y_pred = clf.predict(x_valid_dropna)


###Creating a confusion matrix
cm = confusion_matrix(y_valid_dropna, y_pred)

###Visualizing the confusion matrix
##Labels to represent the encoded class names
labels = ['Non-target', 'Target']

##Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



###Calculating accuray from confusion matrix
accuracy = cm.trace()/cm.sum()

print(accuracy)


###Visualizing feature importance
## Extract the top 15 most important features
importances = pd.Series(clf.feature_importances_, index=x_train.columns)
importance_top_15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(y=range(len(importance_top_15)), width=importance_top_15)# Plot horizontal bar chart
plt.yticks(ticks=range(len(importance_top_15)), labels=importance_top_15.index) # Set y-axis labels
plt.gca().invert_yaxis()  # Show most important feature at the top
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances")
plt.show()

