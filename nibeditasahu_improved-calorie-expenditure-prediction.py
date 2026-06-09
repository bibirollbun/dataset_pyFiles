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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


train_df.shape


test_df.shape


train_df.info()


test_df.info()


plt.figure(figsize = (10,6))
sns.histplot(train_df["Calories"], bins = 50, kde = True, color = "g")
plt.title("Calories Distribution")
plt.show()


plt.figure(figsize = (8,4))
sns.boxplot(x = train_df["Calories"], color = "g")
plt.title("Calories Boxplot")
plt.show()


numeric_df = train_df.select_dtypes(include = ["int", "float"])
corr_matrix = numeric_df.corr()
corr_matrix


plt.figure(figsize = (15, 6))
sns.heatmap(corr_matrix, annot=True, cmap = "prism")
plt.title("Feature Correlation Heatmap")
plt.show()


combined_df = pd.concat([train_df, test_df], axis = 0)
combined_df


combined_encoded_df = pd.get_dummies(combined_df, drop_first=True)


train_encoded_df = combined_encoded_df.iloc[:len(train_df), :]
test_encoded_df = combined_encoded_df.iloc[len(train_df):, :]


features = [col for col in train_encoded_df.columns if col not in \
            ["id", "Calories"]]

X = train_encoded_df[features]
y = train_encoded_df["Calories"]


from sklearn.model_selection import train_test_split


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)


import numpy as np


y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)


from sklearn.ensemble import RandomForestRegressor


RF = RandomForestRegressor(
    n_estimators = 200,
    max_depth = 10,
    random_state = 42,
    n_jobs = -1
)

RF.fit(X_train, y_train_log)


y_pred_log = RF.predict(X_valid)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(0, y_pred) 


from sklearn.metrics import mean_squared_log_error


RMSLE = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print(f"RMSLE: {RMSLE:.4f}")


X_test = test_encoded_df[X_train.columns]
test_pred_log = RF.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(0, test_pred)


submission = pd.DataFrame({
    "id": test_df["id"],
    "Calories": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission Sucessful!")

