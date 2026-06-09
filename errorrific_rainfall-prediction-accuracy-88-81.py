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


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df.isnull().sum()


df = df.drop(columns=["id","day"])


df


df.corr()


X = df.drop(columns=['rainfall'])
y = df['rainfall']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=23, test_size=0.2)


import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score


def objective(trial):
    """Objective function for Optuna to optimize XGBClassifier"""
    params = {
        "n_estimators":trial.suggest_int("n_estimators",100,1000,step=50),
        "learning_rate":trial.suggest_loguniform("learning_rate",0.01,0.3),
        "max_depth": trial.suggest_int("max_depth",3,15),
        "min_child_weight": trial.suggest_int("min_child_weight",1,10),
        "subsample": trial.suggest_float("subsample",0.5,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 0, 5),
        "alpha": trial.suggest_float("alpha", 0, 5),
        "objective": "binary:logistic",  # Use 'multi:softmax' for multi-class classification
        "eval_metric": "logloss",
        "use_label_encoder": False
    }

    # Train XGBClassifier with suggested parameters
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return 1-accuracy

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best parameters:{study.best_params}")


best_params = study.best_params

final_model = xgb.XGBClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy:.4f}")


df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_test.fillna(df_test.mean(), inplace=True)



test_id = df_test['id']
df_test = df_test.drop(columns=['id','day'])
prediction = final_model.predict(df_test)


prob_class_1 = final_model.predict_proba(df_test)[:, 1]


submission = pd.DataFrame({
    "id": test_id, 
    "rainfall": prob_class_1 
})

# Save submission file
submission.to_csv("submission.csv", index=False)


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


X = df.drop(columns = ['rainfall'])
y = df['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),  
    Dense(256, activation='tanh'),
    Dropout(0.3),  
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5, 
    restore_best_weights=True
)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=1, 
                    callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten() 

accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy Score: {accuracy:.4f}")




