import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv',index_col=0)
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv',index_col=0)


train.info()


train.describe()


sampled_train=train.sample(frac=0.05)


from sklearn.model_selection import train_test_split


X=sampled_train.drop('BeatsPerMinute',axis=1)
y=sampled_train['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


def scaler(train_X,test_X):
    m_train,n=train_X.shape
    m_test=test_X.shape[0]
    means=np.zeros(n)
    stds=np.zeros(n)
    scaled_train=np.zeros((m_train,n))
    scaled_test=np.zeros((m_test,n))
    for i in range(n):
        means[i]=np.mean(train_X[:,i])
        stds[i]=np.std(train_X[:,i])
    for i in range(n):
        scaled_train[:,i]=(train_X[:,i]-means[i])/stds[i]
        scaled_test[:,i]=(test_X[:,i]-means[i])/stds[i]
    return scaled_train,scaled_test


scaled_X_train,scaled_X_test=scaler(X_train.to_numpy(),X_test.to_numpy())


def f_wb(X,w,b):
    return np.dot(X,w)+b


def gradient_descent(X,y,max_iter,learning_rate,lambda_):
    m,n=X.shape
    w=np.zeros(n)
    b=0
    for k in range(max_iter):
        dj_dw=np.zeros(n)
        dj_db=0
        for j in range(n):
            for i in range(m):
                dj_dw[j]+=(f_wb(X[i],w,b)-y[i])*X[i,j]
            dj_dw/=m
            dj_dw+=lambda_*w[j]/m
        for i in range(m):
            dj_db+=(f_wb(X[i],w,b)-y[i])
        dj_db/=m
        w=w-dj_dw*learning_rate
        b=b-dj_db*learning_rate
    return w,b


def lin_reg(train_X,train_y,test_X,max_iter=1000,learning_rate=0.1,lambda_=1.0):
    w,b=gradient_descent(train_X,train_y,max_iter,learning_rate,lambda_)
    m,n=test_X.shape
    test_y=np.zeros(m)
    for i in range(m):
        test_y[i]=f_wb(test_X[i],w,b)
    return test_y


pred=lin_reg(scaled_X_train,y_train.to_numpy(),scaled_X_test)


def mae(y_true,y_pred):
    return np.mean(abs(y_true-y_pred))
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


print('MAE :',mae(y_test,pred))
print('MSE :',mse(y_test,pred))
print('RMSE :',mse(y_test,pred)**0.5)


print(mse(y_test,pred)**0.5/np.mean(y_test)*100,'%')


plt.scatter(y_test.to_numpy(),pred)


# Final Submission


scaled_X,scaled_test=scaler(X.to_numpy(),test.to_numpy())


pred=lin_reg(scaled_X,y.to_numpy(),scaled_test)


final=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv',index_col=0)


final['BeatsPerMinute']=pred


final.to_csv('final_linreg.csv')

