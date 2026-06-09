import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


simple_solution = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv")
test = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv")
train = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv")
print(simple_solution.shape)
print(test.shape)
print(train.shape)


test.head()


train.head()


sns.pairplot(train)
plt.show()


sns.barplot(data=train, x='price', y='airline')
plt.show()


train.isnull().sum()


train.info()


X = train.drop('price', axis=1)
y = train[['price']]


train.stops.value_counts()


X.head(1)


def prepared(df):
    df[['fl_country', 'fl_num']] = df['flight'].str.split("-", expand=True)
    df['fl_num'] = df['fl_num'].astype('int64')
    df.drop(['id', 'flight'], axis=1, inplace=True)
    
    ord_encoder = OrdinalEncoder()
    df[['airline', 'fl_country']] = ord_encoder.fit_transform(df[['airline', 'fl_country']])
    
    hot_encoder = ['source_city', 'departure_time', 'arrival_time', 'destination_city']
    df = pd.get_dummies(df, columns=hot_encoder)
    
    stops = {'zero': 0, 'one': 1, 'two_or_more': 2}
    classes = {'Economy': 0, 'Business': 1}
    df['stops'] = df['stops'].replace(stops)
    df['class'] = df['class'].replace(classes)

    std_scaler = StandardScaler()
    df = std_scaler.fit_transform(df)
    
    return df


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


X_train = prepared(X_train)


X_train


X_test = prepared(X_test)


X_test


LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

predict = LR_model.predict(X_test)

MAE = mean_absolute_error(y_test, predict)

print(f"MAE: {MAE}")


LOG_model = LogisticRegression()
LOG_model.fit(X_train, y_train)

predict1 = LOG_model.predict(X_test)

MAE = mean_absolute_error(y_test, predict)

print(f"MAE: {MAE}")


Tree_model = DecisionTreeRegressor()
Tree_model.fit(X_train, y_train)

predict = Tree_model.predict(X_test)

MAE = mean_absolute_error(y_test, predict)

print(f"MAE: {MAE}")


RF_model = RandomForestRegressor(n_estimators=5)
RF_model.fit(X_train, y_train)

predict = RF_model.predict(X_test)

MAE = mean_absolute_error(y_test, predict)

print(f"MAE: {MAE}")


test = prepared(test)


test


prediction = Tree_model.predict(test)


simple_solution['price'] = prediction


simple_solution.head()


simple_solution.to_csv("submission.csv", index=False)




