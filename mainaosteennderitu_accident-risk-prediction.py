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


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df.head()


df.set_index("id",inplace=True)
df.head()


df.info()


#
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown = "ignore",sparse_output = False)
encoded_cat_cols = encoder.fit_transform(df[cat_cols])

encoded_cat_df = pd.DataFrame(encoded_cat_cols,columns=encoder.get_feature_names_out(cat_cols))

df = pd.concat([df[num_cols].reset_index(drop=True),encoded_cat_df.reset_index(drop=True)],axis = 1)
df.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_test.set_index("id",inplace=True)
df_test.head()


cat_test = df_test.select_dtypes(exclude = np.number).columns.tolist()
num_test = df_test.select_dtypes(include = np.number).columns.tolist()

cat_test


cat_test = df_test.select_dtypes(exclude = np.number).columns.tolist()
num_test = df_test.select_dtypes(include = np.number).columns.tolist()

encoded_cat_test = encoder.transform(df_test[cat_test])
encoded_cat_test_df = pd.DataFrame(encoded_cat_test,columns = encoder.get_feature_names_out(cat_test))

df_test = pd.concat([df_test[num_test].reset_index(drop=True),encoded_cat_test_df.reset_index(drop=True)],axis=1)
df_test.head()


df_test.shape


X_train = df.drop("accident_risk",axis=1)
y_train = df["accident_risk"]

X_test = df_test
y_test = y_train.iloc[-172585:]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100,
                              random_state=42,
                              n_jobs = -1,
                              max_depth = None ,
                              max_features = "auto")
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred)

from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model,X_train,y_train,cv=5,scoring = "neg_mean_squared_error")
cv_mse = -np.mean(cv_score)
r2_Score = r2_score(y_test,y_pred)
print(f"CV_MSE: {cv_mse}")



print(f"Test_R2:{r2_Score}")


residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

plt.tight_layout()


print("===SUBMITTED SUCCESSFULLY===")

submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
submission["accident_risk"] = y_pred
submission.to_csv("/kaggle/working/my_submission.csv",index=False)
submission.head()

