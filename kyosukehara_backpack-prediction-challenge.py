# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# å‰�å‡¦ç�†
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# å­¦ç¿’ãƒ¢ãƒ‡ãƒ«
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# è©•ä¾¡
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# complete function
def complete():
    print("\nComplete\n")


# read files 
data = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(data)
complete()


# id columnã‚’å‰Šé™¤
df.drop("id", axis=1, inplace=True)


print("Information: \n{}\n".format(df.info()))
print("Descriptiion: \n{}\n".format(df.describe()))
print("Number of null: \n{}\n".format(df.isnull().sum()))
print("Null percentage desascending: \n{}\n".format(df.isnull().mean().sort_values(ascending=False)))


# ç›®çš„å¤‰æ•°ã�®ç¢ºèª�
sns.histplot(df, x="Price", hue="Brand", multiple='stack', kde=True)


# ãƒ‡ãƒ¼ã‚¿ã�®åˆ†å¸ƒã�®ç¢ºèª�
sns.histplot(df, x="Weight Capacity (kg)", kde=True)


# Weight Capacity ã�«å¯¾ã�—ã�¦ã�®æ¬ æ��å€¤å‡¦ç�†
print(df["Weight Capacity (kg)"].describe())

imputer = SimpleImputer(strategy='mean')
# fit_tranformã�¯NumPyé…�åˆ—ã‚’è¿”ã�™
df["Weight Capacity (kg)"] = imputer.fit_transform(df[["Weight Capacity (kg)"]])

print("Number of Null is: {}".format(df["Weight Capacity (kg)"].isnull().sum()))


# æ¬ æ��å€¤å‡¦ç�†
columns_null = ["Color", "Brand", "Material", "Style", "Laptop Compartment", "Waterproof", "Size"]

# Unknown
for c in columns_null:
    df[c].fillna("Unknown", inplace=True)

print("Complete")


# ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã‚’Numå�‹ã�«å¤‰æ�›
# label_encoding 
# Brand, Material, laptop compartment, waterproof, style, color, 

le = LabelEncoder()
col_to_encode = ["Brand", "Material", "Laptop Compartment", "Waterproof", "Style", "Color", "Size"]

for col in col_to_encode:
    df[col] = le.fit_transform(df[col])

print("\ncomplete\n")


# ãƒ‡ãƒ¼ã‚¿å�‹ã�®ç¢ºèª�ã�¨æ¬ æ��å€¤ã�®ç¢ºèª�
print(df.info())
print(df.isnull().sum())

complete()


# å¤šé‡�å…±ç·šæ€§ã�®ç¢ºèª�

plt.figure(figsize=(30, 30))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)


# split X, and Y
y = df["Price"]
X = df.drop(columns=["Price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("complete")


gbrt = GradientBoostingRegressor(random_state=42, max_depth=2, learning_rate=0.01, n_jobs=-1)
lr = LinearRegression()
rfr = RandomForestRegressor(random_state=42, max_depth = 2, n_jobs=-1)

gbrt.fit(X_train, y_train)
lr.fit(X_train, y_train)
rfr.fit(X_train, y_train)

gbrt_y_pred = gbrt.predict(X_train)
lr_y_pred = lr.predict(X_train)
rfr_y_pred = rfr.predict(X_train)

print("complete")


print("Train RMSE\n")
train_gbrt = gbrt.predict(X_train)
train_lr = lr.predict(X_train)
train_rfr = rfr.predict(X_train)

for pred in [train_gbrt, train_lr, train_rfr]:
    rmse = mean_squared_error(pred, y_train, squared=False)  # squared=False ã�§å¹³æ–¹æ ¹ã‚’å�–ã‚‹
    print(f" \nRMSE: {rmse}")

print("\nTest RMSE\n")

predict_gbrt = gbrt.predict(X_test)
predict_lr = lr.predict(X_test)
predict_rfr = rfr.predict(X_test)

for pred in [predict_gbrt, predict_lr, predict_rfr]:
    rmse = mean_squared_error(pred, y_test, squared=False)  # squared=False ã�§å¹³æ–¹æ ¹ã‚’å�–ã‚‹
    print(f" \nRMSE: {rmse}")


def evaluate_cross_validation(model, x, y, K):
    cv = KFold(K,shuffle=True,random_state=42)
    scores = cross_val_score(model,x,y,cv=cv, scoring="neg_root_mean_squared_error")
    print(-scores)
    print ("Mean score: {} (+/-{}\n Std score: {})".format(np.mean (-scores), sem(-scores), np.std(-scores)))

print(evaluate_cross_validation(gbrt, X_train, y_train, 10))
print(evaluate_cross_validation(lr, X_train, y_train, 10))
print(evaluate_cross_validation(rfr, X_train, y_train, 10))


# laod test data
test_data = "/kaggle/input/playground-series-s5e2/test.csv"
testDf = pd.read_csv(test_data)

# æ¬ æ��å€¤å‡¦ç�†ã�§IDã‚’ä¸€éƒ¨å‰Šé™¤ã�—ã�Ÿã�Ÿã‚�IDã‚’ä¿�ç®¡
test_ids = testDf["id"].copy()

# testãƒ‡ãƒ¼ã‚¿ã�«å‡¦ç�†ã‚’è¡Œã�†
def preprocessing_test(testDf):
    testDf.drop("id", axis=1, inplace=True)
    
    imputer = SimpleImputer(strategy='mean')
    # fit_tranformã�¯NumPyé…�åˆ—ã‚’è¿”ã�™
    testDf["Weight Capacity (kg)"] = imputer.fit_transform(testDf[["Weight Capacity (kg)"]])

    # æ¬ æ��å€¤å‡¦ç�†
    columns_null = ["Color", "Brand", "Material", "Style", "Laptop Compartment", "Waterproof", "Size"]
    
    # Unknown
    for column in columns_null:
        testDf[column].fillna("Unknown", inplace=True)

    le = LabelEncoder()
    col_to_encode = ["Brand", "Material", "Laptop Compartment", "Waterproof", "Style", "Color", "Size"]
    
    for col in col_to_encode:
        testDf[col] = le.fit_transform(testDf[col])

    return testDf

# call preprocessing function 
testdf_encoded = preprocessing_test(testDf)

# call model and fit with entire test data
submission_model = GradientBoostingRegressor(random_state=42, max_depth=2, learning_rate=0.01)

submission_model.fit(X, y)

# predict
predicts = submission_model.predict(testdf_encoded)

submission = pd.DataFrame({"id": test_ids, "Price": np.nan})
submission.loc[testdf_encoded.index, "Price"] = predicts  # äºˆæ¸¬å€¤ã‚’é�©ç”¨

print(submission.describe())

# æ¬ æ��å€¤ã�«ã�¯Mean Valueã‚’ä»£å…¥ã€€ï¼ˆä¸­å¤®å€¤ã�§ã‚‚ã�„ã�„ã�Œï¼‰
submission["Price"].fillna(submission["Price"].median(), inplace=True)

submission.to_csv("BackPack_submission1.csv", index=False)

print("complete")


print(submission.info())

