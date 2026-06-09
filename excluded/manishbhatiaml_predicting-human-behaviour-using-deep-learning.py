!pip install scikeras


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
import warnings
warnings.filterwarnings('ignore')
import keras
from keras import optimizers,regularizers
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense,Dropout
from scikeras.wrappers import KerasClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from tensorflow import random
from keras.callbacks import EarlyStopping




train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv',index_col="id")
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv',index_col="id")
submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


#Checking the Sample submission
submission.head()


print(submission.shape)


#Checking Train and Test contents
#Train
train.head(3)


#Test
test.tail(3)


#Checking Shapes
print(train.shape)
print(test.shape)


#Checking distribution of target data
plt.pie(train['Personality'].value_counts(),labels=train['Personality'].value_counts().keys(),autopct='%1.1f%%',textprops={'fontsize':20,'fontweight':'bold'})
plt.show()


#Checking Missing testues
missing_train=(train.isnull().sum()[train.isnull().sum()>0]).to_frame().rename(columns={0:'No of Missing Values'})
missing_train['% of Missing Values']=round((100*train.isnull().sum()[train.isnull().sum()>0]/len(train)),2)
missing_train.sort_values(by=['% of Missing Values'],ascending=False,inplace=True)
missing_train


missing_test=(test.isnull().sum()[test.isnull().sum()>0]).to_frame().rename(columns={0:'No of Missing Values'})
missing_test['% of Missing Values']=round((100*test.isnull().sum()[test.isnull().sum()>0]/len(test)),2)
missing_test.sort_values(by=['% of Missing Values'],ascending=False,inplace=True)
missing_test


print(train.info())
print("\n")
print("*"*40)
print("\n")
print(test.info())


train.describe()


train.columns


cat_cols=train.select_dtypes(include=['object']).columns.tolist()
cat_cols.pop()
num_cols=train.select_dtypes(include=['number']).columns.tolist()
print(cat_cols)
print(num_cols)


X=train.iloc[:,:-1]
y=train.iloc[:,-1]




print(X.head(5))


print(y.head(5))


#Splitting the train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,shuffle=True)
print(X_train.shape,y_test.shape)
print(X_test.shape,y_test.shape)


le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)



le_name_mapping=dict(zip(le.classes_,le.transform(le.classes_)))
le_name_mapping


print(y_train[:30])


print(y_test[:30])


num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
cat_pipeline=make_pipeline(SimpleImputer(strategy="most_frequent"),OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan))



ct=ColumnTransformer([("num",num_pipeline,num_cols),("cat",cat_pipeline,cat_cols)],verbose_feature_names_out=False,remainder='passthrough').set_output(transform='pandas')


X_train=pd.DataFrame(ct.fit_transform(X_train),columns=ct.get_feature_names_out())
X_test=pd.DataFrame(ct.transform(X_test),columns=ct.get_feature_names_out())



test=pd.DataFrame(ct.transform(test),columns=ct.get_feature_names_out())


print(test.shape)


#Checking the Transformed dataframe shape
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


print(test.shape)


X_train['Stage_fear']=X_train['Stage_fear'].astype(int)
X_test['Stage_fear']=X_test['Stage_fear'].astype(int)
X_train['Drained_after_socializing']=X_train['Drained_after_socializing'].astype(int)
X_test['Drained_after_socializing']=X_test['Drained_after_socializing'].astype(int)



test['Stage_fear']=test['Stage_fear'].astype(int)
test['Drained_after_socializing']=test['Drained_after_socializing'].astype(int)


test.head()


#Checking the contents of Transformed Dataset
X_train.head()


print(X_train.info())
print("\n")
print("*"*40)
print("\n")
print(X_test.info())


#Checking the NUll testues
X_train.isnull().sum()


X_test.isnull().sum()


seed=12
np.random.seed(seed)
random.set_seed(seed)


'''params = {
'model__activation':['relu','tanh'],
'model__optimizer': ['adam','rmsprop','sgd'],
'batch_size':[50,100],
'model__dropout': [0.1,0.2],
'epochs':[100,200]
}
'''


'''
def create_model(activation,optimizer,dropout):
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=128, activation=activation,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dropout(dropout))
    model.add(Dense(units=64, activation=activation,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dropout(dropout))
    model.add(Dense(units=8, activation=activation,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
'''


#model=KerasClassifier(model=create_model)'''


#random_search=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,verbose=1,n_iter=3)'''


#random_search_result=random_search.fit(X_train,y_train)

#print("Best Paramaters",random_search.best_params_)
#print("Best Score",random_search.best_score_)


def final_model():
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], units=256, activation='relu',kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu',kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation='relu',kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


fn_model=final_model()


es_callback = EarlyStopping(monitor='test_loss', \
mode='min', patience=20)


history=fn_model.fit(X_train,y_train,epochs=100,batch_size=100,validation_data=(X_test,y_test),callbacks=[es_callback])


fn_model.evaluate(X_test,y_test)


predictions=fn_model.predict(test)


final_predictions=(predictions > 0.5).astype(int)
submission["Personality"]=le.inverse_transform(final_predictions)
submission.to_csv("submission.csv", index=False)
submission.head()




