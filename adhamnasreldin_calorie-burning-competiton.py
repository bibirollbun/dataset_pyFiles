# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


df.info()


df.drop('id',axis=1, inplace=True)


df['Sex'] = df['Sex'].replace({'male':1,'female':0})


df.tail()


df['BMI']= df['Weight']/df['Height']
df['feat1']= (df['Heart_Rate']*df['Body_Temp'])/df["Duration"]
df['Calories']=df.pop("Calories")
df.tail()


plt.scatter(df['Calories'],df['Duration'])
plt.xlabel('Calories')
plt.show()


plt.hist(df['Calories'],bins=10)
plt.show()


columns=df.columns
corr_matrix = df[columns].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve, confusion_matrix


y= df["Calories"]
x = df.drop("Calories", axis=1)


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



plt.hist(y_test)
plt.show()


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# Predicting the validation set
y_test_pred_lin_reg = linear_model.predict(X_test)



plt.hist(y_test_pred_lin_reg)
plt.show()


y_test_pred_lin_reg = np.abs(y_test_pred_lin_reg)


plt.hist(y_test_pred_lin_reg)
plt.show()


training_score = linear_model.score(X_train, y_train)
print(f"The training coefficent of determination is {training_score}")


from sklearn.metrics import mean_squared_log_error



lin_reg_err= np.sqrt(mean_squared_log_error(y_test, y_test_pred_lin_reg))
print(f"The Root Mean Squared Logarithmic Error for linear regression is {lin_reg_err}")
print(f"The The testing coefficent of determination iss {linear_model.score(X_test,y_test)}")


# Train SVM model
svm_model = SVR(kernel='linear', C=0.7, max_iter=150)  # You can change the kernel to 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)


y_train_pred = svm_model.predict(X_train)
y_test_pred_svm = svm_model.predict(X_test)
svm_model.score(X_train,y_train)


y_test_pred_svm = np.abs(y_test_pred_svm)
svm_err= np.sqrt(mean_squared_log_error(y_test, y_test_pred_svm))
print(f"The Root Mean Squared Logarithmic Error for SVM is {svm_err}")


df.head()


print(f"The Root Mean Squared Logarithmic Error for linear regression is {lin_reg_err}")
print(f"The Root Mean Squared Logarithmic Error for SVM is {svm_err}")


from sklearn.preprocessing import PolynomialFeatures


poly_features = PolynomialFeatures(2)
x2_new = poly_features.fit_transform(x)


xp2_train, xp2_test,yp2_train , yp2_test = train_test_split(x2_new,y,test_size=0.2,random_state=42)


sc2 =StandardScaler()
sc2.fit_transform(xp2_train)
sc2.transform(xp2_test)


polynomial_regression2= LinearRegression()
polynomial_regression2.fit(xp2_train,yp2_train)


 yp2_test_predict =polynomial_regression2.predict(xp2_test)

plt.hist(yp2_test_predict, bins=20)
plt.title("Distribution of polynomial predicted test")
plt.show()


yp2_test_predict= np.maximum(0.75,yp2_test_predict)


poly_reg_err2= np.sqrt(mean_squared_log_error(yp2_test, yp2_test_predict))
print(f"The Root Mean Squared Logarithmic Error for Polynomial regression is {poly_reg_err2}")


poly3_features = PolynomialFeatures(3,interaction_only=True)
x3_new = poly3_features.fit_transform(x)


xp3_train, xp3_test,yp3_train , yp3_test = train_test_split(x3_new,y,test_size=0.2,random_state=42)


sc3 =StandardScaler()
sc3.fit_transform(xp3_train)
sc3.transform(xp3_test)


polynomial_regression3= LinearRegression()
polynomial_regression3.fit(xp3_train,yp3_train)


 yp3_test_predict =polynomial_regression3.predict(xp3_test)

plt.hist(yp3_test_predict, bins=20)
plt.title("Distribution of polynomial predicted test")
plt.show()


yp3_test_predict= np.maximum(0.75,yp3_test_predict)


poly_reg_err3= np.sqrt(mean_squared_log_error(yp3_test, yp3_test_predict))
print(f"The Root Mean Squared Logarithmic Error for Polynomial regression is {poly_reg_err3}")


test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_data.head()


test_data['BMI']= test_data['Weight']/test_data['Height']
test_data['feat1']= (test_data['Heart_Rate']*test_data['Body_Temp'])/test_data["Duration"]
# test_data['Calories']=test_data.pop("Calories")
test_data.tail()


test_data['Sex'] = test_data['Sex'].replace({'male':1,'female':0})


test_id= test_data.drop('id',axis=1,inplace=True)
test_data.describe()


test_trans=poly3_features.transform(test_data)
test_trans_scaled=sc3.transform(test_trans)


sub_pred = polynomial_regression3.predict(test_trans_scaled)
sub_pred = np.maximum(0.5,sub_pred)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = sub_pred
submission.to_csv("submission.csv", index=False)
print('Sumission done!')
submission.head()




