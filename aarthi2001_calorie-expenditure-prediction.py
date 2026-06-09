# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn libraries
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,mean_squared_error

# ignoring harmless warnings
import warnings
warnings.filterwarnings('ignore')


# Training Dataset
train_df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


train_df.head()


train_df.info()


train_df['Sex'].value_counts()


train_df['Sex']=train_df['Sex'].map({'female':'F','male':'M'})
train_df['Sex'].value_counts()


train_df.drop('id',axis=1,inplace=True)


train_df.describe()


plt.pie(train_df['Sex'].value_counts(),labels=train_df['Sex'].value_counts().index,autopct='%1.1f%%')
plt.legend()
plt.show()


for col in train_df.columns:
  if train_df[col].dtype!='O':
    sns.displot(data=train_df,x=col)
    plt.show()





# dependent and independent split

x=train_df.drop('Calories',axis=1)
y=train_df[['Calories']]


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,test_size=0.3)


from sklearn.preprocessing import OneHotEncoder

# training dataset
encoder=OneHotEncoder(sparse_output=False,drop='first',dtype='int')
train_array=encoder.fit_transform(xtrain[['Sex']])
encoded_xtrain=pd.DataFrame(train_array,index=xtrain.index,columns=encoder.get_feature_names_out())

# testing dataset
test_array=encoder.transform(xtest[['Sex']])
encoded_xtest=pd.DataFrame(test_array,index=xtest.index,columns=encoder.get_feature_names_out())


# Training dataset
xtrain=pd.concat([encoded_xtrain,xtrain.drop('Sex',axis=1)],axis=1)

# Testing dataset
xtest=pd.concat([encoded_xtest,xtest.drop('Sex',axis=1)],axis=1)


from sklearn.preprocessing import StandardScaler

# Training Dataset
scaler=StandardScaler()
scaled_xtrain=scaler.fit_transform(xtrain)
final_xtrain=pd.DataFrame(scaled_xtrain,index=xtrain.index,columns=scaler.get_feature_names_out())

# Testing Dataset
scaled_xtest=scaler.transform(xtest)
final_xtest=pd.DataFrame(scaled_xtest,index=xtest.index,columns=scaler.get_feature_names_out())


from sklearn.linear_model import LinearRegression

LRmodel=LinearRegression()
LRmodel.fit(final_xtrain,ytrain)



# Prediction
LRpred=LRmodel.predict(final_xtest)

# Evolution
print("Training Score",LRmodel.score(final_xtrain,ytrain))
print("Training Score / R2 score",r2_score(ytest,LRpred))
print("Mean Absolute Error",mean_absolute_error(ytest,LRpred))
print('Mean Squared Error',mean_squared_error(ytest,LRpred))


from sklearn.ensemble import RandomForestRegressor

RFmodel=RandomForestRegressor(random_state=42,n_estimators=300)
RFmodel.fit(final_xtrain,ytrain)


# Prediction
RFpred=RFmodel.predict(final_xtest)

# Evolution
print("Training Score",RFmodel.score(final_xtrain,ytrain))
print("Training Score / R2 score",r2_score(ytest,RFpred))
print("Mean Absolute Error",mean_absolute_error(ytest,RFpred))
print("Mean Squared Error",mean_squared_error(ytest,RFpred))


test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df.info()


submission=test_df['id']


test_df['Sex']=test_df['Sex'].map({'female':'F','male':'M'})


test_df.drop('id',axis=1,inplace=True)


train_df.head()


# testing dataset
test_array=encoder.transform(test_df[['Sex']])
encoded_xtest=pd.DataFrame(test_array,index=test_df.index,columns=encoder.get_feature_names_out())

# Testing dataset
xtest=pd.concat([encoded_xtest,test_df.drop('Sex',axis=1)],axis=1)


# Testing Dataset
scaled_xtest=scaler.transform(xtest)
final_xtest=pd.DataFrame(scaled_xtest,index=xtest.index,columns=scaler.get_feature_names_out())


RFmodel_test_predict=RFmodel.predict(final_xtest)


submission=pd.concat([submission,pd.DataFrame(RFmodel_test_predict)],axis=1)
submission.rename(columns={0:'Calories'},inplace=True)
submission.head()


submission.to_csv("submission.csv", index=False)

