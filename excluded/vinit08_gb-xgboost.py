import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_df


test_df


submission_df


train_df.info()


test_df.info()


train_df.describe()


test_df.describe()


train_df.isna().sum()


train_df.isna().sum() / train_df.shape[0] *100


test_df.isna().sum()


numeric_cols = [
    "Age" , 	
    "Height" ,	
    "Weight" , 
    "Duration",	
    "Heart_Rate",	
    "Body_Temp",	
    "Calories"
]


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[i], kde=True , color = "green")
    plt.title(f"Histogram of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")



for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(train_df[i], color = "green")
    plt.title(f"Barplot of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")



for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(train_df[i], fill = True)
    plt.title(f"Kde of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")



count = train_df["Sex"].value_counts().sort_values(ascending = False)
# percentage = train_df["Sex"].value_counts() / train_df["Sex"].count() *100

plt.figure(figsize=(6, 6))
plt.pie(count, labels=count.index, autopct='%1.2f%%')
plt.title("Distribution of Sex" , size = 22)
plt.show()



from sklearn.model_selection import train_test_split
new_train_df , val_df =  train_test_split(train_df , test_size = 0.2 , random_state=42)


len(new_train_df) , len(val_df)


new_train_df = new_train_df.dropna()
val_df = val_df.dropna()


input_cols = [
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"
]
target_cols = ["Calories"]


train_inputs = new_train_df[input_cols]
train_target = new_train_df[target_cols]


train_inputs


train_target


val_inputs = val_df[input_cols]
val_target = val_df[target_cols]


val_inputs


val_target


test_inputs = test_df[input_cols]
test_inputs


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


train_inputs["Sex"] = encoder.fit_transform(train_inputs["Sex"])


train_inputs


val_inputs["Sex"] = encoder.fit_transform(val_inputs["Sex"])


val_inputs


test_inputs["Sex"] = encoder.fit_transform(test_inputs["Sex"])


test_inputs


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


copy_train = train_inputs
# copy_train = scaler.fit_transform(copy_train.iloc[1:])


copy_train.iloc[: , 1:] = scaler.fit_transform(copy_train.iloc[: , 1:])


copy_val = val_inputs
# copy_val = scaler.fit_transform(copy_val.iloc[1:])
copy_val.iloc[: , 1:] = scaler.fit_transform(copy_val.iloc[: , 1:])


copy_test = test_inputs
# copy_test = scaler.fit_transform(copy_test.iloc[1:])
copy_test.iloc[: , 1:] = scaler.fit_transform(copy_test.iloc[: , 1:])


import numpy as np
class MeanRegressor():
  def fit(self , inputs , targets):
    self.mean = targets.mean()

  def predict(self , inputs):
    return np.full(inputs.shape[0] , self.mean)


mean_model = MeanRegressor()


mean_model.fit(train_inputs , train_target)


train_preds= mean_model.predict(train_inputs)
train_preds


val_preds = mean_model.predict(val_inputs)
val_preds


from sklearn.metrics import mean_squared_log_error
train_rmsle= mean_squared_log_error(train_target ,train_preds )
train_rmsle


val_rmsle= mean_squared_log_error(val_target ,val_preds )
val_rmsle


from sklearn.linear_model import LinearRegression
linreg_model = LinearRegression()


linreg_model.fit(train_inputs , train_target)


train_preds = linreg_model.predict(train_inputs)
train_preds


val_preds = linreg_model.predict(val_inputs)
val_preds








def evaluate(model):
    train_preds = np.maximum(model.predict(train_inputs), 0)
    train_rmse = mean_squared_log_error(train_target, train_preds)
    val_preds = np.maximum(model.predict(val_inputs), 0)
    val_rmse = mean_squared_log_error(val_target, val_preds)
    return train_rmse, val_rmse, train_preds, val_preds


def predict_and_submit(model, fname):
    test_preds = np.maximum(model.predict(test_inputs), 0)
    sub_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    sub_df['Calories'] = test_preds
    sub_df.to_csv(fname, index=None)
    return sub_df


from sklearn.linear_model import Ridge


model1 = Ridge(random_state = 42)


%%time
model1.fit(train_inputs , train_target)


evaluate(model1)


predict_and_submit(model1 , "ridge_submission.csv")


from sklearn.ensemble import RandomForestRegressor


model2 = RandomForestRegressor(max_depth = 10,
                               n_jobs = -1,
                               random_state = 42,
                               n_estimators = 400
                              )


%%time
model2.fit(train_inputs , train_target)


evaluate(model2)


predict_and_submit(model2 , "rf_submission.csv")


from xgboost import XGBRegressor


model3 = XGBRegressor(random_state=42, n_jobs=-1, 
                      objective='reg:squarederror' , 
                      n_estimators = 300 , max_depth = 10)


model3.fit(train_inputs , train_target)


evaluate(model3)


predict_and_submit(model3 ,"xgb_submission.csv")


import matplotlib.pyplot as plt
def test_params(modelclass , **params):

      model = modelclass(**params).fit(train_inputs , train_target)
      train_rmse = mean_squared_log_error(np.maximum(model.predict(train_inputs), 0) , train_target)
      val_rmse = mean_squared_log_error(np.maximum(model.predict(val_inputs), 0) , val_target)
      return train_rmse , val_rmse
    

def test_param_and_plot(ModelClass, param_name, param_values, **other_params):

    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])

    

 




best_params = {
    "random_state" : 42,
    "n_jobs":-1,
    "objective": "reg:squarederror"
}


test_param_and_plot(XGBRegressor , "n_estimators" , [100 , 200, 500] , **best_params)


best_params["n_estimators"] = 200


%%time
test_param_and_plot(XGBRegressor , "max_depth" , [3,4,5 , 6 , 8 ,10] , **best_params)


best_params["max_depth"] = 5


%%time
test_param_and_plot(XGBRegressor , "learning_rate" , [0.05,0.1 , 0.25 , 0.6 , 0.9] , **best_params)


best_params["learning_rate"] = 0.25


xgb_model_final = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42,
                               n_estimators=200, max_depth=5, learning_rate=0.25,
                               subsample=0.8, colsample_bytree=0.8)


xgb_model_final.fit(train_inputs , train_target)


evaluate(xgb_model_final)


predict_and_submit(xgb_model_final ,"xgbf_submission.csv")


xgb_model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42,
                               n_estimators=200, max_depth=10, learning_rate=0.25,
                               subsample=0.8)


xgb_model.fit(copy_train , train_target)


evaluate(xgb_model)


predict_and_submit(xgb_model ,"xgb_model5_submission.csv")


estimators = [
    ("rf" , RandomForestRegressor(n_estimators=200 , random_state=42),
     ("ridge" , Ridge())),
    ("XG" , XGBRegressor(n_estimators=200 , max_depth = 10 , learning_rate = 0.9))
]


from sklearn.ensemble import StackingRegressor
clf = StackingRegressor(
      estimators=estimators,
      final_estimator=XGBRegressor(),
       cv=20
)


# clf.fit(copy_train , train_target)


evaluate(clf)


predict_and_submit(clf ,"stacking_submission.csv")


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense , Dropout


model = Sequential()

# Input layers
model.add(Dense(40 , activation = "relu" , input_dim = 7))
model.add(Dropout(0.5))

# First hidden layers
model.add(Dense(20 , activation = "relu"))
model.add(Dropout(0.5))

# second hidden layers
model.add(Dense(10 , activation = "relu"))
model.add(Dropout(0.5))

# Third hidden layers
model.add(Dense(5 , activation = "relu"))
model.add(Dropout(0.5))

# output layers
model.add(Dense(1 , activation = "linear"))


model.summary()


from tensorflow.keras.callbacks import EarlyStopping


callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)


model.compile(loss = "mean_squared_error" ,optimizer = "adam" , metrics = ["accuracy"])





history = model.fit(train_inputs , train_target , epochs = 27 , validation_split = 0.2  ,batch_size = 32  ,validation_data = (val_inputs , val_target) ,  callbacks=callback)


import matplotlib.pyplot as plt
plt.plot(history.history["loss"] , label = "lose")
plt.plot(history.history["val_loss"] ,label = "val loss")
plt.legend()


import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"] , label = "accuracy")
plt.plot(history.history["val_accuracy"] ,label = "val accuracy")
plt.legend()


def evaluate1(model):
    train_preds = np.maximum(model.predict(train_inputs), 0)
    train_mse = mean_squared_log_error(train_target, train_preds)
    val_preds = np.maximum(model.predict(val_inputs), 0)
    val_mse = mean_squared_log_error(val_target, val_preds)
    return train_mse, val_mse, train_preds, val_preds


evaluate1(model)


def predict_and_submit1(model, fname):
    test_preds = np.maximum(model.predict(test_inputs), 0)
    sub_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    sub_df['Calories'] = test_preds
    sub_df.to_csv(fname, index=None)
    return sub_df


predict_and_submit1(model , "nn_model4.csv")




