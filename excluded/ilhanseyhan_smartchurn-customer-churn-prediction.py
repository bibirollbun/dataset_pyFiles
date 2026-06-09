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


train_dataset = pd.read_csv("/kaggle/input/streaming-subscription-churn-model/train.csv")
test_dataset = pd.read_csv("/kaggle/input/streaming-subscription-churn-model/test.csv")

train_dataset.head()


train_dataset = train_dataset.drop(columns ="customer_id")
test_dataset = test_dataset.drop(columns = "customer_id")


train_dataset.head()


unique_list = ["subscription_type","payment_plan","payment_method","customer_service_inquiries"]
for i in unique_list:
    print(f"{i}: ",train_dataset[i].unique())


train_dataset["customer_service_inquiries"] = train_dataset["customer_service_inquiries"].replace({"Medium":0,"Low":1,"High":2}).astype(int)
test_dataset["customer_service_inquiries"] = test_dataset["customer_service_inquiries"].replace({"Medium":0,"Low":1,"High":2}).astype(int)

train_dataset["payment_plan"] = train_dataset["payment_plan"].replace({"Yearly":0,"Monthly":1}).astype(int)
test_dataset["payment_plan"] = test_dataset["payment_plan"].replace({"Yearly":0,"Monthly":1}).astype(int)

train_dataset = pd.get_dummies(train_dataset,columns = ["subscription_type","payment_method","location"])
test_dataset = pd.get_dummies(test_dataset,columns = ["subscription_type","payment_method","location"])



train_dataset.info()


from sklearn.model_selection import train_test_split

x = train_dataset.drop(columns = "churned")
y = train_dataset["churned"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,roc_auc_score

model_random = RandomForestClassifier(n_estimators = 100)
model_random.fit(x_train,y_train)
y_pred_random = model_random.predict(x_test)
print("Classification Report",classification_report(y_test,y_pred_random))
print("AUC SCORE: " ,roc_auc_score(y_test,y_pred_random))


from lightgbm import LGBMClassifier as LGBM

model_lgbm = LGBM(n_estimators = 100)
model_lgbm.fit(x_train,y_train)
y_pred_lgbm = model_lgbm.predict(x_test)
print("Classification Report",classification_report(y_test,y_pred_lgbm))
print("AUC SCORE: " ,roc_auc_score(y_test,y_pred_lgbm))


import xgboost as xgb

model_xgb = xgb.XGBClassifier(verbosity=2)
model_xgb.fit(x_train,y_train)
y_pred_xgb = model_xgb.predict(x_test)

print("Classification Report",classification_report(y_test,y_pred_xgb))
print("AUC SCORE: " ,roc_auc_score(y_test,y_pred_xgb))


model_list = {
    "Random Forest": model_random,
    "lightgbm": model_lgbm,
    "xgboost": model_xgb    
}

for name,model in model_list.items():
    y_pred = model.predict(x_test)
    print(f"{name}\n")
    print("Classification Report",classification_report(y_test,y_pred))
    print("AUC SCORE: " ,roc_auc_score(y_test,y_pred))


from lightgbm import LGBMClassifier as LGBM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

update_model_gbm = LGBM(random_state = 0)

param_grid = {
    "n_estimators" : [100,200],
    "max_depth" : [6,8],
    "learning_rate": [0.01,0.1],
    "num_leaves": [20,31],
    "min_child_samples" : [20,30],
}



grid_search = GridSearchCV(
    estimator = update_model_gbm,
    param_grid = param_grid,
    scoring = "roc_auc",
    cv = 3,
    verbose = 2,
    n_jobs=-1
)


grid_search.fit(x_train,y_train)
print("en iyi sonuç: ",grid_search.best_score_)
print("en iyi değerler: ",grid_search.best_params_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:,1]
print(classification_report(y_test,y_pred))
print("AUC Değeri: ",roc_auc_score(y_test,y_proba))


import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = pd.Series(best_model.feature_importances_,index = x_train.columns)
feature_importances.nlargest(15).sort_values().plot(kind ="barh",figsize = (6,8),title ="Özellik Önlemleri")
plt.grid()


import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test,plot_type="bar")


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaler = sc.fit_transform(x_train)
x_test_scaler = sc.transform(x_test)
test_dataset_scaler = sc.transform(test_dataset)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential([
    Dense(64,input_shape=(x_train_scaler.shape[1],),activation = "relu"),
    Dropout(0.3),
    Dense(32,activation="relu"),
    Dropout(0.3),
    Dense(1,activation="sigmoid")    
])

model.compile(
    optimizer="adam",
    loss ="binary_crossentropy",
    metrics = ["accuracy",tf.keras.metrics.AUC()]
)

history = model.fit(
    x_train_scaler,y_train,
    epochs = 30,
    batch_size = 256,
    validation_split=0.2,
    verbose = 2
)

loss, accuracy,auc =model.evaluate(x_test_scaler,y_test)
print("Test AUC: ",auc)
print("Test Accuracy: ",accuracy)


from sklearn.metrics import confusion_matrix,classification_report
y_pred_prop = model.predict(x_test_scaler)
y_pred = (y_pred_prop > 0.5).astype(int)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

fgr,tpr, thresholds = roc_curve(y_test,y_pred_prop)
roc_auc = auc(fgr,tpr)

plt.figure(figsize = (6,5))
plt.plot(fgr,tpr,label="AUC = %0.2f" %roc_auc)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()


y_submission_proba = model.predict(test_dataset_scaler)
y_submission = (y_submission_proba>0.5).astype(int).reshape(-1)

test_csv = pd.read_csv("/kaggle/input/streaming-subscription-churn-model/test.csv")  

submission = pd.DataFrame({
    "customer_id": test_csv["customer_id"],  # Orijinal test dosyasındaki ID sütunu
    "prediction": y_submission
})

submission.to_csv("submission.csv",index = False)




