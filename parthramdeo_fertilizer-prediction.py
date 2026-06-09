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


import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv") 
test  = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.sample(5)


test.sample(5)


train_id = train["id"]
test_id  =  test["id"]
train=train.drop("id",axis=1)
test =test.drop("id",axis=1)


train.sample(5)


test.sample(5)


train['Soil Type'].value_counts().plot(kind="bar")


test['Soil Type'].value_counts().plot(kind="bar")


train.isnull().sum()


Na_list = ["missing", "N/A", "null"]

for col in train.columns:
    count = train[col].isin(Na_list).sum()
    print(f"{col}: {count} missing entries")



import matplotlib.pyplot as plt
import seaborn as sns


for i in train.columns:
    if train[i].dtype != "object":
        sns.kdeplot(data=train,x=f"{i}")
        plt.show()


sns.pairplot(train)


train["Fertilizer Name"].value_counts().plot(kind="pie")


train["Crop Type"].value_counts().plot(kind="pie")


train["Soil Type"].value_counts().plot(kind="pie")


train.describe()


for i in train.columns:
    if train[i].dtype !="object":
        sns.boxplot(data=train,x=f"{i}",palette="pastel")
        plt.show()
    


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(max_depth=10,n_estimators=100)
X = train.iloc[:,:-1]
y = train.iloc[:,-1]

X = pd.get_dummies(X,drop_first=True)
le = LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
rf.fit(X_train,y_train)

importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)


top_n = 20
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df.head(top_n),
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title('Top Feature Importances from RandomForest')
plt.tight_layout()
plt.show()


num_cols = train[[col for col in train.columns if train[col].dtype != "object"]]
num_cols.head()


X_train_num,X_test_num,y_train_num,y_test_num = train_test_split(num_cols,y,test_size=0.2,random_state=42)


#from sklearn.metrics import accuracy_score
#rf_num = RandomForestClassifier(n_estimators=100,random_state=42)
#rf_num.fit(X_train_num,y_train_num)
#y_pred_rf_num=rf_num.predict(X_test_num)
#y_prob_rf_num = rf_n#um.predict_proba(X_test_num)
#accuracy_score(y_test_num,y_pred_rf_num)


def mapk(actual, prob, k=3):
    topk = np.argsort(prob, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for a, preds in zip(actual, topk):
        try:
            rank = np.where(preds == a)[0][0] + 1
            score += 1.0 / rank
        except IndexError:
            pass
    return score / len(actual)

#print(f"The mean average precision score of top 3 predictions : {mapk(y_test_num,y_prob_rf_num)}")


#sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
#sample_sub.sample(5)


#import optuna

#def objective(trial):
    # Suggest values for hyperparameters
  # # n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
  #  max_depth = trial.suggest_int("max_depth", 2, 32,64)
  #  min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
  #  min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
#
  #  # Create and fit random forest model
  #  model = RandomForestClassifier(
  #  n_estimators=n_estimators,
  #  max_depth=max_depth,
  #  min_samples_split=min_samples_split,
  #  min_samples_leaf=min_samples_leaf,
  #  random_state=42,
  #  )
  #  model.fit(X_train_num, y_train_num)
#
  #  # Make predictions and calculate MAP
  #  y_prob_num = model.predict_proba(X_test_num)
  #  map3_score = mapk(y_test_num,y_prob_num)
  #  return map3_score


# Create study object
#study = optuna.create_study(direction="maximize")

# Run optimization process
#study.optimize(objective, n_trials=20, show_progress_bar=True)


# Print best trial and best hyperparameters
#print("Best trial:", study.best_trial)
#print("Best hyperparameters:", study.best_params)


#train["Fertilizer Name"].value_counts()


#params = {'n_estimators': 14, 'max_depth': 2, 'min_samples_split': 10, 'min_samples_leaf': 1}
#rf_final = RandomForestClassifier(**params)
#rf_final.fit(X_train_num, y_train_num)
test_df = pd.concat([test_id, test[num_cols.columns]], axis=1)


# 1. Get predicted probabilities
#probs = rf_final#.predict_proba(test_df[num_cols.columns])  # shape: (n_samples, n_classes)
#
## 2. Get indices of top 3 classes for each row
#top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)
#
## 3. Convert indices back to class labels
## If using LabelEncoder
#top3_labels = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
#
## 4. Join labels into space-separated strings
#top3_strings = [' '.join(row) for row in top3_labels]


#submission = pd.DataFrame()
#submission["id"] = test_id


#submission.shape


#len(top3_strings)


#submission["Fertilizer Name"] = top3_strings


#submission.to_csv("/kaggle/working/submission.csv", index=False)


#import tensorflow 
#from tensorflow import keras
#from keras.layers import Dense
#from keras.models import Sequential 
#import keras_tuner as kt


#def build_model(hp):
#    model = Sequential()
#    optimizers = hp.Choice("optimizer",["adam","RMSprop","SGD"])
#    activations = hp.Choice("activation",["relu","elu","tanh"])
#    
#    num_layers = hp.Int("number-hidden-layers",max_value=5,min_value=2)
#    num_neurons = hp.Int("number-neurons-per-layers",max_value=128,min_value=8,step=8)
#    for i in range(num_layers):
#        if i==0:
#            model.add(Dense(num_neurons,input_dim=6,activation=activations))
#        else:
#            model.add(Dense(num_neurons,activation=activations))
#    model.add(Dense(7,activation="softmax"))
#    model.compile(optimizer=optimizers,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#    return model


#tuner = kt.RandomSearch(
#    build_model,objective="val_accuracy",max_trials=10,executions_per_trial=2,directory="mydir",project_name="mytuned2"
#)


#tuner.search(X_train_num,y_train_num,validation_data=(X_test_num,y_test_num),epochs=5)


#np.unique(np.array(y_test_num))


#tuner.get_best_hyperparameters()[0].values


#model = tuner.get_best_models()[0]


#model.summary()


#model.fit(
#    X_train_num,
#    y_train_num,
#    epochs=50,
#    validation_data=(X_test_num, y_test_num),
#    initial_epoch=5
#)
#


#test_df = pd.concat([test_id, test[num_cols.columns]], axis=1)



# 1. Get predicted probabilities
#probs = model.predict(test_df[num_cols.columns])  # shape: (n_samples, n_classes)
#
## 2. Get indices of top 3 classes for each row
#top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)
#
## 3. Convert indices back to class labels
## If using LabelEncoder
#top3_labels = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
#
## 4. Join labels into space-separated strings
#top3_strings = [' '.join(row) for row in top3_labels]


#submission["Fertilizer Name"] = top3_strings
#submission.to_csv("/kaggle/working/submission.csv", index=False)


train.head()


train["Fertilizer Name"].value_counts().plot(kind="bar")


train.info()


for cols in train.columns:
    if train[cols].dtype!="object" or train[cols].dtype!="bool":
        plt.figure(figsize=(10,6),dpi=150)
        plt.hist(x=train[cols],bins=20)
        plt.title(f"content of {cols} in soil / measure of env condition")
        plt.xlabel(f"{cols}")
        plt.show()
        


train.sample(10)


le_soil_type = LabelEncoder()
le_crop_type = LabelEncoder()

train["Soil Type"] = le_soil_type.fit_transform(train["Soil Type"])
test["Soil Type"] = le_soil_type.transform(test["Soil Type"])
train["Crop Type"] = le_crop_type.fit_transform(train["Crop Type"])
test["Crop Type"] = le_crop_type.transform(test["Crop Type"])

from xgboost import XGBClassifier

model = XGBClassifier(
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    learning_rate=0.01,
    max_depth=8,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

le_target = LabelEncoder()
train["Fertilizer Name"] = le_target.fit_transform(train["Fertilizer Name"])

X_train,X_test,y_train,y_test = train_test_split(train[[col for col in train.columns if col!="Fertilizer Name"]],train["Fertilizer Name"],test_size=0.2,random_state=42)

model.fit(X_train, y_train)





from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))



y_prob = model.predict_proba(X_test)
top_3 = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1]


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    `actual` is a list of true labels.
    `predicted` is a list of predicted label lists (top k predictions).
    """
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                score += 1.0 / (i + 1)
                break
        return score

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



score_map3 = mapk(y_test, top_3, k=3)
print(f"MAP@3 score: {score_map3:.5f}")


Submission_Predicts = model.predict_proba(test)


## 2. Get indices of top 3 classes for each row
top3_indices = np.argsort(Submission_Predicts, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)

 #3. Convert indices back to class labels
# If using LabelEncoder
top3_labels = le_target.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)

## 4. Join labels into space-separated strings
top3_strings = [' '.join(row) for row in top3_labels]


submission = pd.DataFrame()
submission["id"] = test_id
submission["Fertilizer Name"] = top3_strings


submission.to_csv("/kaggle/working/submission.csv", index=False)


train["temp_humid"] = train["Temparature"]*train["Humidity"]
train["temp_moist"] = train["Temparature"]*train["Moisture"]
train["moist_humid"] = train["Moisture"]*train["Humidity"]

test["temp_humid"] = test["Temparature"]*test["Humidity"]
test["temp_moist"] = test["Temparature"]*test["Moisture"]
test["moist_humid"] = test["Moisture"]*test["Humidity"]




model2 = XGBClassifier(
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    learning_rate=0.01,
    max_depth=8,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


X_train,X_test,y_train,y_test = train_test_split(train[[col for col in train.columns if col!="Fertilizer Name"]],train["Fertilizer Name"],test_size=0.2,random_state=42)

model2.fit(X_train, y_train)


y_pred = model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))



y_prob = model2.predict_proba(X_test)
top_3 = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1]


score_map3 = mapk(y_test, top_3, k=3)
print(f"MAP@3 score: {score_map3:.5f}")


train["temp_humid"] = train["Temparature"]*train["Humidity"]
train["temp_moist"] = train["Temparature"]*train["Moisture"]
train["moist_humid"] = train["Moisture"]*train["Humidity"]

test["temp_humid"] = test["Temparature"]*test["Humidity"]
test["temp_moist"] = test["Temparature"]*test["Moisture"]
test["moist_humid"] = test["Moisture"]*test["Humidity"]

train.drop(["temp_humid","temp_moist","moist_humid"],axis=1,inplace=True)
test.drop(["temp_humid","temp_moist","moist_humid"],axis=1,inplace=True)

train["temp_humid_moist"] = train["Temparature"]*train["Humidity"]*train["Moisture"]
test["temp_humid_moist"] = test["Temparature"]*test["Humidity"]*test["Moisture"]
train["const"] = 0
test["const"] = 0


model3 = XGBClassifier(
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    learning_rate=0.01,
    max_depth=10,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


X_train,X_test,y_train,y_test = train_test_split(train[[col for col in train.columns if col!="Fertilizer Name"]],train["Fertilizer Name"],test_size=0.2,random_state=42)

model3.fit(X_train, y_train)


y_pred = model3.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))


y_prob = model3.predict_proba(X_test)
top_3 = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1]


score_map3 = mapk(y_test, top_3, k=3)
print(f"MAP@3 score: {score_map3:.5f}")


Submission_Predicts = model3.predict_proba(test)


## 2. Get indices of top 3 classes for each row
top3_indices = np.argsort(Submission_Predicts, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)

 #3. Convert indices back to class labels
# If using LabelEncoder
top3_labels = le_target.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)

## 4. Join labels into space-separated strings
top3_strings = [' '.join(row) for row in top3_labels]


submission = pd.DataFrame()
submission["id"] = test_id
submission["Fertilizer Name"] = top3_strings


submission.to_csv("/kaggle/working/submission.csv", index=False)




