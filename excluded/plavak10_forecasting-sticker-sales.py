import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings(action="ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.head(10)


train = train_df.copy()
train.info()


train = train.dropna(subset=['num_sold'])
train.info()


train.drop('id',axis=1,inplace=True)
train.head()


#Convert date column to datetime format
train['date'] = pd.to_datetime(train['date'])

#Create 2 separate columns for month and year
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

#Drop date column
train.drop('date',axis=1,inplace=True)
train = train.reset_index(drop=True)

train.head()


train


for column in ['country','store','product','month','year']:
    plt.figsize=(6,4)
    sns.countplot(x=column, data=train)
    plt.title(f'Countplot for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(drop='first',sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(train[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
train = pd.concat([train, one_hot_df], axis=1)

# Drop the original categorical columns
train.drop(categorical_columns, axis=1,inplace=True)

train.head()


plt.figsize=(10,8)
sns.heatmap(train.corr())
plt.show()


X = train.drop('num_sold',axis=1)
y = train['num_sold']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


scaler = StandardScaler()

scaler.fit(X_train)

#scale the year column
X_train['year'] = scaler.fit_transform(X_train[['year']])
X_test['year'] = scaler.transform(X_test[['year']])

X_train['month'] = scaler.fit_transform(X_train[['month']])
X_test['month'] = scaler.transform(X_test[['month']])


X_train


X_test


models = {
    "Linear Regression":LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Neural Net": MLPRegressor(),
    "SVM Linear": LinearSVR()
}


for name,model in models.items():
    model.fit(X_train,y_train)
    print(name+" R2 score - {:.2f}".format(model.score(X_test,y_test)))


test_df


test_df.info()


test = test_df.copy()


test['date'] = pd.to_datetime(test['date'])

#Create 2 separate columns for month and year
test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year

#Drop date column
test.drop('date',axis=1,inplace=True)
test = test.reset_index(drop=True)

test


categorical_columns = test.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(drop='first',sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(test[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
test = pd.concat([test, one_hot_df], axis=1)

# Drop the original categorical columns
test.drop(categorical_columns, axis=1,inplace=True)

test.head()


test['year'] = scaler.fit_transform(test[['year']])
test['month'] = scaler.fit_transform(test[['month']])

test


test.drop('id',axis=1,inplace=True)
test


knn = KNeighborsRegressor()

knn.fit(X_train,y_train)

ans = knn.predict(test)
ans


ans = np.round(ans,0)
ans


new_df = pd.DataFrame({'id':test_df['id'],'num_sold':ans})
new_df


new_df.to_csv('/kaggle/working/output.csv',index=False)

