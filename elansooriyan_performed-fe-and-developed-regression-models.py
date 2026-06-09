import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',None)


data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
data.head()


test_data = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


na_features = []
for col in data.columns:
    if data[col].isnull().any():
        na_features.append(col)
na_features



na_numerical_features = []
na_categorical_features = []

for col in na_features:
    if data[col].dtype == 'O':
        na_categorical_features.append(col)
    else:
        na_numerical_features.append(col)

na_categorical_features
na_numerical_features


#We are imputing for age feature
data['Age_mean'] = data['Age'].fillna(data['Age'].mean())


plt.figure()
data['Age'].plot(kind='kde')
data.Age_mean.plot(kind='kde',color='red')


data = data.drop('Age_mean',axis=1)


for col in na_numerical_features:
    if (data[col].skew() <= 0.5) and (data[col].skew() >= -0.5) :
        data[col] = data[col].fillna(data[col].mean())
        test_data[col] = test_data[col].fillna(test_data[col].mean())
    else:
        data[col] = data[col].fillna(data[col].median())
        test_data[col] = test_data[col].fillna(test_data[col].median())
        


data[na_numerical_features].isnull().sum()


for col in na_categorical_features:
    data[col] = data[col].fillna(data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])


data[na_categorical_features].isnull().sum()


#train dataset
data['Previous Claims'] = np.log1p(data['Previous Claims'])
data['Premium Amount'] = np.log1p(data['Premium Amount'])
data['Annual Income'] = np.log(data['Annual Income'])

#test dataset
test_data['Previous Claims'] = np.log1p(test_data['Previous Claims'])
test_data['Annual Income'] = np.log(test_data['Annual Income'])



categorical_features = data.select_dtypes(include='object')
categorical_features.head()


#train data
OHE_data = pd.get_dummies(data,columns = ['Marital Status','Occupation','Location','Property Type'],prefix=['Marital Status','Occupation','Location','Property Type'])
#test data
OHE_test_data = pd.get_dummies(test_data,columns = ['Marital Status','Occupation','Location','Property Type'],prefix=['Marital Status','Occupation','Location','Property Type'])


OHE_data.head()


from sklearn.preprocessing import OrdinalEncoder

#mentioning order for ordinal relationship
categories = [
    ['High School',"Bachelor's","Master's",'PhD'],
    ['Basic', 'Comprehensive', 'Premium'],
    ['Poor', 'Average', 'Good'],
    ['Rarely', 'Monthly', 'Weekly', 'Daily']
    ]

ordinal_encoder = OrdinalEncoder(categories=categories)
#train data
OHE_data[['Education Level','Policy Type', 'Customer Feedback', 'Exercise Frequency']] = ordinal_encoder.fit_transform(OHE_data[['Education Level','Policy Type', 'Customer Feedback', 'Exercise Frequency']])

#test data
OHE_test_data[['Education Level','Policy Type', 'Customer Feedback', 'Exercise Frequency']] = ordinal_encoder.fit_transform(OHE_test_data[['Education Level','Policy Type', 'Customer Feedback', 'Exercise Frequency']])


#train data
OHE_data['Smoking Status'] = np.where(OHE_data['Smoking Status'] == 'Yes',1,0)
OHE_data['Gender'] = np.where(OHE_data['Gender'] == 'Male',1,0)

#test data
OHE_test_data['Smoking Status'] = np.where(OHE_test_data['Smoking Status'] == 'Yes',1,0)
OHE_test_data['Gender'] = np.where(OHE_test_data['Gender'] == 'Male',1,0)


OHE_data.head()


from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

#train data
OHE_data[['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']] = scalar.fit_transform(OHE_data[['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']])

#test data
OHE_test_data[['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']] = scalar.fit_transform(OHE_test_data[['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']])
OHE_data.head()


y_train = OHE_data['Premium Amount']
x_train = OHE_data.drop(['Premium Amount','id','Policy Start Date'],axis=1)
x_train['Year'] = OHE_data['Policy Start Date'].str[0:4].astype(int)
x_train['Month'] = OHE_data['Policy Start Date'].str[5:7].astype(int)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_train)


from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error


# Evaluation Metrics
mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


mae = mean_absolute_error(y_train, y_pred)
regression_accuracy = 1 - (mae / np.mean(y_train))
regression_accuracy


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_train)



mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


mae = mean_absolute_error(y_train, y_pred)
regression_accuracy = 1 - (mae / np.mean(y_train))
regression_accuracy


x_test = OHE_test_data.drop(['id','Policy Start Date'],axis=1)
x_test['Year'] = OHE_data['Policy Start Date'].str[0:4].astype(int)
x_test['Month'] = OHE_data['Policy Start Date'].str[5:7].astype(int)


y_test=model.predict(x_test)


submit = pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")
submit["Premium Amount"] = np.exp(y_test)-1
submit.to_csv("submission.csv",index=False)

