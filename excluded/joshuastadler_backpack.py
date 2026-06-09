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


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train.info()


test.info()


numerical_feats =["Compartments","Weight Capacity (kg)"]
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_train = train.copy()
imputed_test = test.copy()
imputed_extra = extra.copy()
for feats in numerical_feats:
    imputed_train[feats] = my_imputer.fit_transform(train[[feats]])
    train[feats] = imputed_train[feats]
    imputed_test[feats] = my_imputer.fit_transform(test[[feats]])
    test[feats] = imputed_test[feats]
    imputed_extra[feats] = my_imputer.fit_transform(extra[[feats]])
    extra[feats] = imputed_extra[feats]
    



for column in train.columns:
    print(column, 'options', train[column].unique())

category_feats = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for feat in category_feats:
    train[feat].fillna("Unknown")
    test[feat].fillna("Unknown")
    extra[feat].fillna("Unknown")


for col in category_feats:
    value_counts = train[col].value_counts()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(6,2,i+1)

    sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
    
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Number of Each Category in {col}", fontsize=14)
    plt.xticks(rotation=45)
    
    plt.show()


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for feat in category_feats:
    train[feat] = label_encoder.fit_transform(train[feat])
    test[feat] = label_encoder.fit_transform(test[feat])
    extra[feat] = label_encoder.fit_transform(extra[feat])


train.info()


test.info()


extra.info()


X_train = train.drop(columns=["Price", "id",'Color'])
y_train = train["Price"]
X_test = test.drop(columns=["id",'Color'])
X_extra = extra.drop(columns =["Price", "id",'Color'])
y_extra = extra["Price"]


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,12))
for i, col in enumerate(category_feats):
    plt.subplot(4,2,i+1)
    sns.boxplot(x=train[col], y=train["Price"], palette="coolwarm")
    plt.title(f"Price Distribution by {col}")
plt.tight_layout()
plt.show()



from xgboost import XGBRegressor
my_model = XGBRegressor(
    learning_rate=0.01,#rate of learning to prevent overfitting
    n_estimators=638,  # number of estimators 
    random_state=42 #fixed randomness
)
my_model.fit(X_train,y_train)
# my_model.fit(
#     X_train, y_train,
#     eval_set=[(X_extra, y_extra)],
#     eval_metric="mae",  # Using Mean Absolute Error (MAE) as the evaluation metric
#     early_stopping_rounds=50,  # Stops training if no improvement after 50 rounds
#     verbose=True
# )
# best_n_estimators = my_model.best_iteration
# print(f"Optimal n_estimators: {best_n_estimators}")


from sklearn.metrics import mean_absolute_error
predictions=my_model.predict(X_extra)
print(mean_absolute_error(predictions, y_extra))


y_test_predict = my_model.predict(X_test)
print(y_test_predict)


submission = pd.DataFrame({
    "id": test['id'],
    'Price': y_test_predict
})

submission.to_csv('submission.csv', index=False)


