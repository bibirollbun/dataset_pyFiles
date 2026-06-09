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


df1 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")
df2 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv")
df3 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")
df4 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv")
sample_submission = pd.read_csv("/kaggle/input/zillow-prize-1/sample_submission.csv")


df1.head(10)


df2.head(10)


print(df4.columns)
print(df3.columns)


df2= pd.merge(df2,df1, on='parcelid', how='left')
df4 = pd.merge(df4, df3, on='parcelid', how='left')


X = pd.concat([df2,df4], axis=0)


df1.shape


df3.shape


X.shape


X = X.dropna(subset=["logerror"])
y = X["logerror"]


X = X.drop(columns=["logerror"])


numeric_features = X.select_dtypes(include=["int64","float64"]).columns
X = X[numeric_features].copy()


X = X.fillna(X.median())


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.0005)
}



for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"{name} RMSE = {rmse}")



final_model = Ridge(alpha=1.0)
final_model.fit(X, y)


test_df = df3.copy()


test_df = test_df[numeric_features].copy()


test_df = test_df.fillna(X.median())


test_preds = final_model.predict(test_df)


submission = pd.DataFrame({
    "ParcelId": df3["parcelid"],
    "logerror": test_preds
})


submission.to_csv("submission.csv", index=False)
print("submission.csv ready!")


submission.head()


import os

for root, dirs, files in os.walk("/kaggle/input", topdown=False):
    for name in files:
        print(os.path.join(root, name))


# Load your current submission file
sub = pd.read_csv("/kaggle/working/submission.csv")

# FIX: Use "logerror" which is the correct column name from the previous step
pred = sub["logerror"] 

# Required Zillow submission columns
required_months = ["201610", "201611", "201612", "201710", "201711", "201712"]

# Create final df
final = pd.DataFrame()
final["ParcelId"] = sub["ParcelId"]

# Add required months
for m in required_months:
    # Since only 'ParcelId' and 'logerror' exist, the 'else' block will run for all months.
    if m in sub.columns:
        final[m] = sub[m]
    else:
        final[m] = pred # This now correctly uses the 'logerror' values

# Save final file
final.to_csv("/kaggle/working/submission.csv", index=False)

final.head()


final.columns




