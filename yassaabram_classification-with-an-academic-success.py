# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Input, Dense, Dropout ,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint ,CSVLogger,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall, F1Score, AUC


df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
df.head()


df.info()


df.drop(columns=["id"],inplace=True)


df.duplicated().sum()


x=df.drop(columns=['Target'])
y=df['Target']


x.head()


#encoder=LabelEncoder()
#y=encoder.fit_transform(y)


encoderr=OneHotEncoder(sparse_output=False)
y=encoderr.fit_transform(y.values.reshape(-1,1))


y


scaler=StandardScaler()
x=scaler.fit_transform(x)


x_train,x_dammy,y_train,y_dammy =train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


x_valid,x_test,y_valid,y_test=train_test_split(x_dammy,y_dammy,test_size=0.5,random_state=42,stratify=y_dammy)


model=Sequential([ Dense(512,activation='relu',input_dim=(x_train.shape[1])),
    Dropout(0.1),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(3,activation="softmax")              
])
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', Precision()]
)
model.summary()


x_train.shape[1]


inp=Input((x_train.shape[1],))
d1=Dense(512,activation='relu')(inp)
dr=Dropout(0.1)(d1)
d2=Dense(256,activation='relu')(dr)
d3=Dense(128,activation='relu')(d2)
out=Dense(3,activation="softmax")(d3)
model=Model([inp],[out])


model.summary()


model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy',Precision()])


modelcheckpoints=ModelCheckpoint('model.weights.keras',monitor='val_loss',save_best_only=True,save_weights_only=False)
earlyStopping=EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True)
logger=CSVLogger('model.csv')


hist=model.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=20,batch_size=32,callbacks=[modelcheckpoints,earlyStopping,logger]) 


hist.history['accuracy']


hist.history['val_accuracy']


tr_loss=hist.history['loss']
vall_loss=hist.history['val_loss']
tr_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
epochs=[i+1 for i in range(len(tr_loss))]

plt.figure(figsize=(16,16))
plt.subplot(1,2,1)
plt.plot(epochs,tr_loss,color='green',label='tr_loss')
plt.plot(epochs,vall_loss,color='red',label='vall_loss')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs,tr_acc,color='green',label='tr_acc')
plt.plot(epochs,val_acc,color='red',label='val_acc')
plt.title('Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend

plt.tight_layout()
plt.show





#model.save('model2.h5')


from tensorflow.keras.models import load_model
model2=load_model('/kaggle/working/model.weights.keras')


model2.evaluate(x_valid,y_valid)


inp=Input((x_train.shape[1],))
d1=Dense(512,activation='relu')(inp)
dr=Dropout(0.1)(d1)
d2=Dense(256,activation='relu')(dr)
dr1=Dropout(0.1)(d2)
d3=Dense(128,activation='relu')(dr1)
out=Dense(3,activation="softmax")(d3)
model2=Model([inp],[out])
model2.summary()


model2.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy',Precision()])


hist2=model2.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=20,batch_size=32,callbacks=[modelcheckpoints,earlyStopping,logger]) 


tr_loss=hist2.history['loss']
vall_loss=hist2.history['val_loss']
tr_acc=hist2.history['accuracy']
val_acc=hist2.history['val_accuracy']
epochs=[i+1 for i in range(len(tr_loss))]

plt.figure(figsize=(16,16))
plt.subplot(1,2,1)
plt.plot(epochs,tr_loss,color='green',label='tr_loss')
plt.plot(epochs,vall_loss,color='red',label='vall_loss')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs,tr_acc,color='green',label='tr_acc')
plt.plot(epochs,val_acc,color='red',label='val_acc')
plt.title('Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend

plt.tight_layout()
plt.show


inp=Input((x_train.shape[1],))
d1=Dense(512,activation='relu')(inp)
BatchNormalization(),
dr=Dropout(0.2)(d1)
d2=Dense(512,activation='relu')(dr)
dr1=Dropout(0.2)(d2)
d6=Dense(512,activation='relu')(dr1)
dr2=Dropout(0.1)(d6)
d3=Dense(256,activation='relu')(dr2)
d4=Dense(128,activation='relu')(d3)
d5=Dense(64,activation='relu')(d4)
out=Dense(3,activation="softmax")(d5)
model3=Model([inp],[out])
model3.summary()


model3.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy',Precision()])


modelcheckpoints=ModelCheckpoint('model.weights.keras',monitor='val_loss',save_best_only=True,save_weights_only=False)
earlyStopping=EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)
logger=CSVLogger('model.csv')


hist3=model3.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=50,batch_size=64,callbacks=[modelcheckpoints,earlyStopping,logger]) 


import matplotlib.pyplot as plt



tr_loss=hist3.history['loss']
vall_loss=hist3.history['val_loss']
tr_acc=hist3.history['accuracy']
val_acc=hist3.history['val_accuracy']
epochs=[i+1 for i in range(len(tr_loss))]

plt.figure(figsize=(16,16))
plt.subplot(1,2,1)
plt.plot(epochs,tr_loss,color='green',label='tr_loss')
plt.plot(epochs,vall_loss,color='red',label='vall_loss')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs,tr_acc,color='green',label='tr_acc')
plt.plot(epochs,val_acc,color='red',label='val_acc')
plt.title('Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend

plt.tight_layout()
plt.show


inp=Input((x_train.shape[1],))
d1=Dense(512,activation='relu')(inp)
BatchNormalization(),
dr=Dropout(0.3)(d1)
d2=Dense(512,activation='relu')(dr)
dr1=Dropout(0.3)(d2)
d6=Dense(512,activation='relu')(dr1)
dr2=Dropout(0.1)(d6)
d3=Dense(256,activation='relu')(dr2)
d4=Dense(128,activation='relu')(d3)
dr3=Dropout(0.1)(d4)
d5=Dense(64,activation='relu')(dr3)
out=Dense(3,activation="softmax")(d5)
model4=Model([inp],[out])
model4.summary()


modelcheckpoints=ModelCheckpoint('model.weights.keras',monitor='val_loss',save_best_only=True,save_weights_only=False)
earlyStopping=EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True)
logger=CSVLogger('model.csv')
reduce=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=7)


model4.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy',Precision()])


hist4=model4.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=35,batch_size=64,callbacks=[modelcheckpoints,reduce,logger]) 


tr_loss=hist4.history['loss']
vall_loss=hist4.history['val_loss']
tr_acc=hist4.history['accuracy']
val_acc=hist4.history['val_accuracy']
epochs=[i+1 for i in range(len(tr_loss))]

plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.plot(epochs,tr_loss,color='green',label='tr_loss')
plt.plot(epochs,vall_loss,color='red',label='vall_loss')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs,tr_acc,color='green',label='tr_acc')
plt.plot(epochs,val_acc,color='red',label='val_acc')
plt.title('Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend

plt.tight_layout()
plt.show


xx=df.drop(columns=['Target'])
yy=df['Target']


encoder=LabelEncoder()
yy=encoder.fit_transform(yy)


scaler=StandardScaler()
xx=scaler.fit_transform(xx)


xxtrain,xxtest,yytrain,yytest=train_test_split(xx,yy,test_size=0.2,random_state=42,stratify=y)


rf=RandomForestClassifier(n_estimators=100,max_depth=9)
rf.fit(xxtrain,yytrain)


print(rf.score(xxtrain,yytrain))
print(rf.score(xxtest,yytest))


xgb=XGBClassifier(n_estimators=25)
xgb.fit(xxtrain,yytrain)


print(xgb.score(xxtrain,yytrain))
print(xgb.score(xxtest,yytest))


lgbm=LGBMClassifier(n_estimators=150)
lgbm.fit(xxtrain,yytrain)


print(lgbm.score(xxtrain,yytrain))
print(lgbm.score(xxtest,yytest))


testdf=pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
testdf.head()


testdfsc=scaler.transform(testdf.iloc[:,1:])



pesd1=lgbm.predict(testdfsc)
pesd1


pesd1=encoder.inverse_transform(pesd1)
pesd1


supp=pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


supp.head()


supp['Target']=pesd1
supp.head()


supp.to_csv('submission.csv',index=False)


pesd=model4.predict(testdfsc)
pesd


pesd=encoderr.inverse_transform(pesd)
pesd


supp=pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


supp['Target'] = pesd.ravel()
supp.head()


supp.to_csv('submission1.csv',index=False)




