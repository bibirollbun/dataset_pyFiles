import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from random import*
#Extracting all data
df= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_extra= pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df= pd.concat([df,df_extra])
test= pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
#The first 5 training example
print(df.head())
print(df.shape)
#Overall Data Description and Stats
CATEGORICAL=["Brand","Material","Size","Laptop Compartment","Waterproof","Style","Color"]
#General price distribution for each feature
fig,ax= plt.subplots(1,6,figsize=(60,10))
for i in range(6):
    a=choice(CATEGORICAL)
    sns.boxplot(data=df,x=a,y="Price",ax=ax[i])
#In depth price distribution for each feature pair
plt.savefig("feature_boxplot.png")
fig,ax= plt.subplots(2,3,figsize=(30,10))
for i in range(2):
    for j in range(3):
        while True:
            a=choice(CATEGORICAL)
            b=choice(CATEGORICAL)
            if a!=b: break
        sns.boxplot(data=df,x=a,y="Price",hue=b,ax=ax[i][j])
plt.savefig("feature_pair_boxplot.png")
print(df.groupby("Waterproof").mean(numeric_only=True))
df.pop("id")
ids= test.pop("id")
#Splitting the data into train and test sets
X_train,X_test= train_test_split(df,test_size=0.2)
print(f"X_train.shape:{X_train.shape}, X_test.shape:{X_test.shape}")


my_learner_params= {"num_trees":400,
                 "max_depth":32,
                 "growing_strategy":"BEST_FIRST_GLOBAL",
                 "compute_oob_variable_importances":True,
                 "winner_take_all": False}
print(tfdf.__version__)
model= tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)




model.compile(loss="mse")


X_train= tfdf.keras.pd_dataframe_to_tf_dataset(X_train,label="Price",task=tfdf.keras.Task.REGRESSION)
X_test= tfdf.keras.pd_dataframe_to_tf_dataset(X_test,label="Price",task=tfdf.keras.Task.REGRESSION)
test= tfdf.keras.pd_dataframe_to_tf_dataset(test,task=tfdf.keras.Task.REGRESSION)
train_data= model.fit(X_train,max_depth=24,num_trees=500)
train_rmse= train_data.history["rmse"]
print(f"Train RMSE:{train_rmse}")



print(np.sqrt(model.evaluate(X_test)))
print(model.summary())



y_hat=model.predict(test)
print(y_hat.shape)
y_hat= y_hat.reshape([200000])
y_hat= list(y_hat)
ids= list(ids)
my_submission= {"id":ids,"Price":y_hat}
my_submission= pd.DataFrame(my_submission)
print(my_submission)
my_submission.to_csv("/kaggle/working/submission.csv",index=False)

