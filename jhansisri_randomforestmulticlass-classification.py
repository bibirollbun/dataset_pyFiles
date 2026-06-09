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


import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df_train = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")


df_train.head()


df_train.info()



df_train.isnull().sum()


target = "NObeyesdad"
X = df_train.drop(columns=[target])
y = df_train[target]


cat_cols = [col for col in X.select_dtypes("O").columns]
num_cols = [col for col in X.columns if col not in cat_cols and col != "id"]


print("CATEGORICAL", cat_cols)


print("Numeric:", num_cols)


for col in cat_cols:
    print(col)
    df_train[col].value_counts(normalize = True).plot(kind = "bar")
    plt.show()


for col in num_cols:
    print(col)
    df_train[col].plot(kind = "hist")
    plt.show()


num_transformer = StandardScaler()  # Scale numerical features

cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # Label encode categorical

# Combine transformers
preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])




pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"Model Accuracy: {accuracy:.2f}")


df_test.head()


list(cat_cols).extend(list(num_cols))


df_pred = pipeline.predict(df_test.iloc[:,1:])


df_pred


submission = pd.DataFrame({
    "id": df_test["id"],  # Assigning unique IDs
    "NObeyesdad": df_pred
})

# **Save to CSV**
submission.to_csv("submission.csv", index=False)



submission.head()


sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e2/sample_submission.csv")


sample_submission.head()




