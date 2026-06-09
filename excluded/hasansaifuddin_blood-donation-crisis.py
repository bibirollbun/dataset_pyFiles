import pandas as pd
import numpy as np
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


class Regression:
    def __init__(self,beta,learning_rate,lasso,ridge,epochs):
        self.learning_rate = learning_rate
        self.lasso = lasso
        self.ridge = ridge
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.beta = beta
        
    def fit(self,X,Y):
        samples,features = X.shape
        self.weights = np.zeros(features)
        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            gradient_W = (-2 * X.T @ (Y-y_pred)) + (self.lasso * np.sign(self.weights)) + (2 * self.ridge * self.weights)
            gradient_b = -2 * np.sum(Y-y_pred)
            gradient_W = np.clip(gradient_W , -1,1)
            gradient_b = np.clip(gradient_b ,-1,1)
            self.weights -= self.learning_rate * gradient_W
            self.bias -=self.learning_rate * gradient_b
            self.lasso = max(0,self.lasso - self.beta * np.sum(np.abs(self.weights)))
            self.ridge = max(0,self.ridge - self.beta * np.sum(self.weights **2))
            if epoch % 500 == 0 or epoch == self.epochs-1:
                loss = np.mean((Y-y_pred)**2)
                avg_weight_change = np.mean(np.abs(gradient_W))
                print(f"Epoch {epoch}:")
                print(f"   Avg Gradient W: {avg_weight_change:.6f}")
                print(f"   Avg Weight: {np.mean(np.abs(self.weights)):.6f}")
                print(f"   Bias: {self.bias}")
                print(f"   位1: {self.lasso:.6f}, 位2: {self.ridge:.6f}")
                mse = mean_squared_error(Y,y_pred)
                print(f"MSE = {mse}")
                print("-" * 50)
    def predict(self,X_test):
        answers = []
        for x_test in X_test:
            predict = x_test @ self.weights + self.bias
            answers.append(predict)
        return np.array(answers)


path = '/kaggle/input/equity-post-HCT-survival-predictions'


train = pd.DataFrame(pd.read_csv(path+'/train.csv'))
test = pd.DataFrame(pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv'))
data_description = pd.DataFrame(pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'))


train = train.drop(columns='ID')



mean = train["efs_time"].mean() 
mean
train = train[train['efs_time']<35]
train.efs_time = train.efs_time /10


train



data_dictionary = pd.DataFrame(pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'))


cat = train.select_dtypes('object').fillna('0')
num = train.select_dtypes('float64').fillna(0)


cat


num


couts = (num['age_at_hct'] >= 65)
couts.sum()



num = num.fillna(0)


for x in cat.columns:
    column = cat[x]
    unique = column.unique()
    i = 0
    for value in unique:
        cat[x] = cat[x].replace(value,i)
        i = i+1


# this is a mapping dictionary
'''
for x in cat.columns:
    column = cat[x]
    unique = column.unique()
    mapp = {values:index for index,values in enumerate(unique)}
    cat[x] = cat[x].map(mapp)
    '''


train = pd.concat([cat,num],axis=1)


train


train = train.astype(float)
x = train.drop(columns=['efs_time'])
y = train.efs_time
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)


model = Regression(0.00025,0.0003,0.1,0.1,7500)


model.fit(x_train,y_train)


x_test = np.array(x_test)


prediction = model.predict(x_test)


accuracy = r2_score(y_test,prediction)


accuracy


prediction






















