# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_data.shape


train_data.isna().sum()


import matplotlib.pyplot as plt
subset_brand = train_data.Brand.value_counts().reset_index()
subset_brand.columns = ['Brand', 'Count']
mean_price = train_data.groupby('Brand')['Price'].mean().reset_index()
mean_price.columns = ['Brand', 'Mean_Price']
mean_price['Mean_Price'] = mean_price['Mean_Price'].round(2)

subset_brand = subset_brand.merge(mean_price, on='Brand', how='inner')

#plt.bar(subset_brand.Brand, subset_brand.Count)
plt.bar(subset_brand.Brand, subset_brand.Mean_Price)
plt.xlabel('Brand')
plt.ylabel('Count')
plt.title('Brand Distribution')
plt.xticks(rotation=45)  
plt.show()


categorical_col = train_data.select_dtypes(include='object').columns
numerical_col = train_data.select_dtypes(exclude='object').columns

numerical_col = numerical_col.drop('Price')
numerical_col = numerical_col.drop('id')
numerical_col


{col: train_data[col].nunique() for col in categorical_col}


for col in categorical_col:
    train_data[col] = train_data[col].fillna('Unknown')
    test_data[col] = train_data[col].fillna('Unknown')


train_data['Weight Capacity (kg)'] = train_data['Weight Capacity (kg)'].fillna(train_data['Weight Capacity (kg)'].mean())
test_data['Weight Capacity (kg)'] = train_data['Weight Capacity (kg)'].fillna(test_data['Weight Capacity (kg)'].mean())


X = train_data[['Material', 'Size',  'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]
Y = train_data['Price']

categorical_features = ['Material', 'Size', 'Laptop Compartment',
       'Waterproof', 'Style']
numerical_features = ['Weight Capacity (kg)']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state =42, test_size =0.2)



encode_categorical_data = OneHotEncoder( handle_unknown='ignore')

X_train_encoded = encode_categorical_data.fit_transform(X_train[categorical_features])
X_test_encoded = encode_categorical_data.transform(X_test[categorical_features])
test_data_encoded = encode_categorical_data.transform(test_data[categorical_features])

X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), 
                               columns=encode_categorical_data.get_feature_names_out(),
                               index=X_train.index).astype(int)

X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), 
                              columns=encode_categorical_data.get_feature_names_out(),
                              index=X_test.index).astype(int)


test_data_encoded = pd.DataFrame(test_data_encoded.toarray(), 
                              columns=encode_categorical_data.get_feature_names_out(),
                              index=test_data.index).astype(int)

X_train_encoded = pd.concat([X_train[numerical_features], X_train_encoded], axis=1)
X_test_encoded = pd.concat([X_test[numerical_features], X_test_encoded], axis=1)
test_data_encoded = pd.concat([test_data[numerical_features], test_data_encoded], axis=1)


model = XGBRegressor(n_estimators=50, n_jobs =4)
scores = cross_val_score(model, X_train_encoded, Y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(scores)


model.fit(X_train_encoded,Y_train)
Y_predict = model.predict(X_test_encoded)
print("XGB Regressor:")

print(f'Mean Absolute Error: {mean_absolute_error(Y_test, Y_predict)}')
print(f'R2-Score: {r2_score(Y_test, Y_predict)}')


from sklearn.ensemble import RandomForestRegressor 
rfg_model = RandomForestRegressor(n_estimators=50, n_jobs=4)


scores = cross_val_score(rfg_model, X_train_encoded, Y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(scores)


rfg_model.fit(X_train_encoded,Y_train)
Y_predict = rfg_model.predict(X_test_encoded)

feature_importance = pd.DataFrame({'Feature': X_train_encoded.columns, 'Importance': rfg_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Random Forest Regressor:")
print(f'Mean Absolute Error: {mean_absolute_error(Y_test, Y_predict)}')
print(f'R2-Score: {r2_score(Y_test, Y_predict)}')

print(feature_importance.head(10))


for i in range(1,3):
    Ridge_model = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('polynomial', PolynomialFeatures(degree=i)),
            ('ridge', Ridge(alpha=0.01))
        ]
    )

    scores = cross_val_score(Ridge_model, X_train_encoded, Y_train, cv=5, scoring=make_scorer(mean_absolute_error))
    print(scores)
    Ridge_model.fit(X_train_encoded,Y_train)
    Y_predict = Ridge_model.predict(X_test_encoded)
    print("RIDGE MODEL:")
    
    print(f'Polynomial Degree {i}')
    print(f'Mean Absolute Error: {mean_absolute_error(Y_test, Y_predict)}')
    print(f'R2-Score: {r2_score(Y_test, Y_predict)}')



test_data_predict = (Ridge_model.predict(test_data_encoded))


output = pd.DataFrame({'id': test_data.id, 'Price': test_data_predict})
output.to_csv("submission.csv", index=False)

# Display first 10 rows
output.head(10)




