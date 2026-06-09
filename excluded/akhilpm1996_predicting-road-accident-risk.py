import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df.head()


df.info()


df.isnull().sum()


df.drop('id',axis = 1,inplace=True)


df.describe()


X = df.drop('accident_risk',axis=1)
y = df['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


category_features = X_train.select_dtypes(include=["object" , "bool"]).columns
numeric_features = X_train.select_dtypes(exclude=["object" , "bool"]).columns

train_cat = X_train[category_features]
train_num = X_train[numeric_features]

test_cat = X_test[category_features]
test_num = X_test[numeric_features]


encoder = OneHotEncoder(sparse_output=False,drop='first')
encoded_train = encoder.fit_transform(train_cat)
encoded_test = encoder.transform(test_cat)
encoded_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out())
encoded_df_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())


final_train = pd.concat([encoded_df.reset_index(drop=True), train_num.reset_index(drop=True)], axis=1)
final_test = pd.concat([encoded_df_test.reset_index(drop=True), test_num.reset_index(drop=True)], axis=1)


model = RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(final_train, y_train)


y_pred = model.predict(final_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


fulldf_cat = X[category_features]
fulldf_num = X[numeric_features]


encoded_data = encoder.transform(fulldf_cat)
encoded_fulldf = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

final_fulldf = pd.concat([encoded_fulldf.reset_index(drop=True), fulldf_num.reset_index(drop=True)], axis=1)


model.fit(final_fulldf, y)


testdata = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
testdata.head()


df = testdata.copy()
df.drop('id',axis=1,inplace=True)
category_features_test = df.select_dtypes(include=["object" , "bool"])
numeric_features_test = df.select_dtypes(exclude=["object" , "bool"])
encoded_test = encoder.transform(category_features_test)

encoded_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())
final_test = pd.concat([encoded_df.reset_index(drop=True), numeric_features_test.reset_index(drop=True)], axis=1)

y_pred = model.predict(final_test)

sub = pd.DataFrame({'id':testdata['id'],'accident_risk':y_pred})
sub.to_csv("submission.csv",index=False)


sub

