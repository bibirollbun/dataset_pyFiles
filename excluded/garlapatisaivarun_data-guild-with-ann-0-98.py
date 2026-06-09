# Importing all necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load Training data
train_df=pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/train.csv")
# train_df.drop(["ID"],inplace=True,axis=1)
train_df.head(2)


# Load Test data
test_df=pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/test.csv")
test_df.head(2)


train_df.shape,test_df.shape


train_df.describe()


train_df.dtypes


train_df.isna().sum()


test_df.isna().sum()


train_df['productive_day'].value_counts()


train_df['hours_of_meetings'] = train_df['hours_of_meetings'].fillna(train_df['hours_of_meetings'].mean())
test_df['hours_of_meetings'] = test_df['hours_of_meetings'].fillna(test_df['hours_of_meetings'].mean())
train_df['hours_of_sleep'] = train_df['hours_of_sleep'].fillna(train_df['hours_of_sleep'].mean())
test_df['hours_of_sleep'] = test_df['hours_of_sleep'].fillna(test_df['hours_of_sleep'].mean())
print(train_df.isna().sum().sum())
print(test_df.isna().sum().sum())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 20))
plt.suptitle("Univariate Analysis", fontsize=20)

columns = ['coffee_intake', 'hours_of_meetings', 'bugs_assigned',
           'lines_of_code_written', 'hours_of_sleep', 'browsing_time']

for i, col in enumerate(columns, 1):
    plt.subplot(6, 2, i)
    sns.histplot(x=col, data=train_df, kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()



plt.figure(figsize=(12,12)).suptitle("Bivariate Analysis of Categorical Variables ",fontsize=20)
plt.subplot(3,1, 1)
sns.boxplot(x='coffee_intake',y='bugs_assigned',data=train_df)
plt.subplot(3,1, 2)
sns.boxplot(y='bugs_assigned',x='productive_day',data=train_df)
plt.subplot(3,1, 3)
sns.boxplot(y='coffee_intake',x='productive_day',data=train_df)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

df=train_df[['coffee_intake', 'hours_of_meetings', 'bugs_assigned',
       'lines_of_code_written', 'hours_of_sleep', 'browsing_time',
       'productive_day']]
correlation_matrix = df.corr()
plt.figure(figsize=(10,5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


train_df=train_df[['coffee_intake', 'hours_of_meetings', 'bugs_assigned',
       'lines_of_code_written', 'hours_of_sleep', 'browsing_time',
       'productive_day']]


X=train_df.iloc[:,0:-1]
y=train_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  


len(X_train),len(X_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime


(X_train.shape[1],)


## Build Our ANN Model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## HL1 Connected wwith input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]

)


model.summary()


import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()
loss


## compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])


## Set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)


### Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[early_stopping_callback]
)


sample=test_df[['coffee_intake', 'hours_of_meetings', 'bugs_assigned',
       'lines_of_code_written', 'hours_of_sleep', 'browsing_time']]
sample = scaler.transform(sample) 


y_pred_test = model.predict(sample)


def round_off(value):
    if value < 0.5 :
        return 0
    else:
        return 1


submission=pd.DataFrame()
sample_df=pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/sample_submission.csv")
submission['Id']=sample_df['Id']
submission['productive_day']=y_pred_test
submission['productive_day']=submission['productive_day'].apply(round_off)
submission.head(5)


# submission.to_csv('/kaggle/working/submission_ANN.csv',index=False)

