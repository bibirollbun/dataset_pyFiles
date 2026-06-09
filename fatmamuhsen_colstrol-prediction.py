import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/kaggle/input/ndsc-regression-cholesterol-prediction/train.csv')


data


data.info()


data.isnull().sum()


data.duplicated().sum()


data.nunique()


data.describe()


data['Jenis Kelamin'].value_counts().reset_index()


sns.countplot(x = 'Jenis Kelamin', data=data)
plt.show()


data['Tempat lahir'].value_counts().nlargest(70).reset_index()


values = data['Tempat lahir'].value_counts().nlargest(70)
index = data['Tempat lahir'].value_counts().nlargest(70).index


plt.figure(figsize = (10,8))
plt.pie(values,labels = index,autopct='%0.2f%%')
plt.show()


le = LabelEncoder()
data['Jenis Kelamin'] = le.fit_transform(data['Jenis Kelamin'])
data['Tempat lahir'] =  le.fit_transform(data['Tempat lahir'])


correlation = data.corr().round(2)


plt.figure(figsize = (10,7))
sns.heatmap(correlation,annot = True)
plt.show()


plt.figure(figsize = (9,6))
sns.heatmap(correlation[['Cholesterol Total (mg/dL)']].sort_values(by = ['Cholesterol Total (mg/dL)'],ascending = False),annot = True)
plt.show()



x = data.drop('Cholesterol Total (mg/dL)',axis=1)
y = data['Cholesterol Total (mg/dL)']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model_1 = LinearRegression()
model_1.fit(x_train , y_train) 


y_train_pred = model_1.predict(x_train)
y_test_pred = model_1.predict(x_test)

mse_train =  np.sqrt(mean_squared_error(y_train, y_train_pred))
mse_test =  np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)
print("R-squared (Train):", r2_train)
print("R-squared (Test):", r2_test)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)


x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly, y, test_size=0.3, random_state=42)


scaler = MinMaxScaler()
x_train_poly = scaler.fit_transform(x_train_poly)
x_test_poly = scaler.transform(x_test_poly)


# Create a polynomial regression model (Linear Regression with polynomial features)
model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train_poly)


# Make predictions
y_train_pred_poly = model_poly.predict(x_train_poly)
y_test_pred_poly = model_poly.predict(x_test_poly)

# Evaluate the model
mse_train_poly =  np.sqrt(mean_squared_error(y_train_poly, y_train_pred_poly))
mse_test_poly = np.sqrt( mean_squared_error(y_test_poly, y_test_pred_poly))

r2_train_poly = r2_score(y_train_poly, y_train_pred_poly)
r2_test_poly = r2_score(y_test_poly, y_test_pred_poly)

print("Polynomial Regression (Degree 2)")
print("Mean Squared Error (Train):", mse_train_poly)
print("Mean Squared Error (Test):", mse_test_poly)
print("R-squared (Train):", r2_train_poly)
print("R-squared (Test):", r2_test_poly)


from sklearn.linear_model import Lasso


# You can adjust the 'alpha' parameter (regularization strength)
lasso_model = Lasso(alpha=0.1) # Example alpha value

# Fit the Lasso model on the polynomial features (using the scaled training data)
lasso_model.fit(x_train_poly, y_train_poly)


# Make predictions using the Lasso model
y_train_pred_lasso = lasso_model.predict(x_train_poly)
y_test_pred_lasso = lasso_model.predict(x_test_poly)

# Evaluate the Lasso model
mse_train_lasso =  np.sqrt(mean_squared_error(y_train_poly, y_train_pred_lasso))
mse_test_lasso =  np.sqrt(mean_squared_error(y_test_poly, y_test_pred_lasso))

r2_train_lasso = r2_score(y_train_poly, y_train_pred_lasso)
r2_test_lasso = r2_score(y_test_poly, y_test_pred_lasso)

print("\nLasso Regression (Polynomial Features, Alpha=0.1)")
print("Mean Squared Error (Train):", mse_train_lasso)
print("Mean Squared Error (Test):", mse_test_lasso)
print("R-squared (Train):", r2_train_lasso)
print("R-squared (Test):", r2_test_lasso)


from sklearn.linear_model import Ridge


# You can adjust the 'alpha' parameter (regularization strength)
ridge_model = Ridge(alpha=1.0) # Example alpha value

# Fit the Ridge model on the polynomial features (using the scaled training data)
ridge_model.fit(x_train_poly, y_train_poly)



# Make predictions using the Ridge model
y_train_pred_ridge = ridge_model.predict(x_train_poly)
y_test_pred_ridge = ridge_model.predict(x_test_poly)

# Evaluate the Ridge model
mse_train_ridge =  np.sqrt(mean_squared_error(y_train_poly, y_train_pred_ridge))
mse_test_ridge =  np.sqrt(mean_squared_error(y_test_poly, y_test_pred_ridge))

r2_train_ridge = r2_score(y_train_poly, y_train_pred_ridge)
r2_test_ridge = r2_score(y_test_poly, y_test_pred_ridge)

print("\nRidge Regression (Polynomial Features, Alpha=1.0)")
print("Root Mean Squared Error (Train):", mse_train_ridge)
print("Mean Squared Error (Test):", mse_test_ridge)
print("R-squared (Train):", r2_train_ridge)
print("R-squared (Test):", r2_test_ridge)




