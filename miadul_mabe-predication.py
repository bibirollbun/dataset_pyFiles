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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("/kaggle/input/MABe-mouse-behavior-detection/train.csv")
df.head()



df.shape            # row, column
df.info()           # data types
df.describe()       # statistics
df.isnull().sum()   # missing values
df.nunique()        # unique values



df = df.fillna(df.mode().iloc[0])




from sklearn.preprocessing import LabelEncoder

# LabelEncoder object
le = LabelEncoder()

# find all categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# apply label encoding to each column
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df.head()




X = df.drop("tracking_method", axis=1)
y = df["tracking_method"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)



y_pred = model.predict(X_val)



print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, cmap="Blues")
plt.show()




df.head()




import pandas as pd

test = pd.read_csv("/kaggle/input/MABe-mouse-behavior-detection/test.csv")
test.head()








































