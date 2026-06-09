import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers,callbacks

import math
import warnings
warnings.filterwarnings('ignore')
# Load the data
def info_data(data):
    print(f"Data Shape: {data.shape}")
    print(f"information about data: {data.info()}")
    print(f"Data Description: {data.describe()}")
    print(f"checking the missing value: {data.isnull().sum()}")
    # plotting the null values according to the columns
    plt.figure(figsize=(10,5))
    missing_data = data.isnull().sum().reset_index()
    if missing_data[0].sum() == 0:
        print("No Missing Values")  
    else:
        px.bar( x=missing_data.index, y=missing_data[0].values ,title='Missing Values in Columns')
        plt.show()
        
def data_clean(data):
    total_data=data.shape[0]
    missing_data_percnt = data.isnull().mean() * 100
    if missing_data_percnt.max() <= 5:
        print("Missing values are less than 5%. So, we can drop the missing values")
        data.dropna(inplace=True)
    for i in data.columns:
        if len(data[i].unique())>5:
            data[i].fillna(data[i].mode().iloc[0], inplace=True)
        elif (data[i].skew()<0) or  (data[i].skew()>0):
            data[i].fillna(data[i].median(),inplace=True)
        else:
            data[i].fillna(data[i].mean(),inplace=True)
    return data   
data=pd.read_csv(r"/kaggle/input/dataset/train.csv")
info_data(data)  
data_cleaned = data_clean(data)
df_train=data_cleaned.sample(frac=0.80,random_state=0)
df_valid=data_cleaned.drop(df_train.index)
X_train=df_train.drop(['rainfall','id','day'],axis=1)
X_valid=df_valid.drop(['rainfall','id','day'],axis=1)
y_train=df_train['rainfall']
y_valid=df_valid['rainfall']
input_shape=X_train.shape[1]
model=keras.Sequential([
    layers.BatchNormalization(input_shape=[input_shape]),
    layers.Dense(128,activation='relu',input_shape=[input_shape]),
    layers.Dropout(rate=0.2),
    layers.BatchNormalization(),
    layers.Dense(128,activation='relu',),
    layers.Dropout(rate=0.2),
    layers.BatchNormalization(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(rate=0.2),
    layers.BatchNormalization(),
    layers.Dense(1,activation='sigmoid')
])
# compiling network
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
early_stopping=callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True
)
### fitting the model
history = model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    epochs=500,
    callbacks=[early_stopping],
    verbose=1
)
history_df=pd.DataFrame(history.history)
history_df.loc[:,['binary_accuracy','val_binary_accuracy']].plot()
history_df.loc[:,['loss','val_loss']].plot()


##Predicting rainfall


test_data=pd.read_csv(r"/kaggle/input/dataset/test.csv")
test_data['winddirection']=test_data['winddirection'].fillna(test_data['winddirection'].mean())
X_test=test_data.drop(['id','day'],axis=1)
y_test_pred_prob = model.predict(X_test)
threshold = 0.5
y_test_pred = (y_test_pred_prob >= threshold).astype(int)
print("Predicted Rainfall for test data (first 10):", y_test_pred[:10].flatten())
output=pd.DataFrame({'id':test_data['id'],'rainfall':y_test_pred.flatten()})



output.shape


test_data.shape




