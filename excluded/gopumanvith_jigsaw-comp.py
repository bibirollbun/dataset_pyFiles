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


train_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


train_data.head()


# train_data["text"] = train_data['body'] + ','+ train_data['rule'] + ','+ train_data['subreddit'] + train_data['positive_example_1'] + ', '+ train_data['positive_example_2'] + ', '+ train_data['negative_example_1'] + ', '+ train_data['negative_example_2'] 
# test_data["text"] = test_data['body'] + ','+ test_data['rule'] + ','+ test_data['subreddit']+ test_data['positive_example_1'] + ', '+ test_data['positive_example_2'] + ', '+ test_data['negative_example_1'] + ', '+ test_data['negative_example_2'] 

train_data["text"] = train_data['body'] + ','+ train_data['rule'] + ','+ train_data['subreddit']
test_data["text"] = test_data['body'] + ','+ test_data['rule'] + ','+ test_data['subreddit'] 


train_data.head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,2))
X = tfidf.fit_transform(train_data["text"])
X_test = tfidf.transform(test_data["text"])
y = train_data["rule_violation"]



X = X.toarray().astype('float32')
X_test = X_test.toarray().astype('float32')



X.shape


train_data.shape


# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation = 'sigmoid')
])


model.compile(
    optimizer = "adam",
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
)


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    restore_best_weights = True,
)


model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs = 10,
    callbacks = [early_stopping],
    verbose =1
)


X_test.shape


pred_nn = model.predict(X_test)
print(pred_nn)


# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error


# rf_model = RandomForestRegressor(max_leaf_nodes = 500, random_state = 42)


# # fit your model
# rf_model.fit(X_train, y_train)


# # Calculate the mean absolute error of your Random Forest model on the validation data
# val_predictions = rf_model.predict(X_val) 
# pred_new = (val_predictions > 0.5).astype(int)
# rf_val_mae = mean_absolute_error(y_val, pred_new)

# print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# rf_model_2 = RandomForestClassifier(
#     n_estimators=100, 
#     max_depth=10,
#     random_state=42)


# # fit your model
# rf_model_2.fit(X_train, y_train)


# # Calculate the mean absolute error of your Random Forest model on the validation data
# val_predictions_2 = rf_model_2.predict(X_val) 
# pred_new_2 = (val_predictions_2 > 0.5).astype(int)
# rf_val_mae_2 = mean_absolute_error(y_val, pred_new_2)

# print("Validation MAE for Random Forest Model: {}".format(rf_val_mae_2))


# xgb_model = xgb.XGBClassifier(
#     n_estimators=100, 
#     learning_rate=0.1, 
#     max_depth=3, 
#     use_label_encoder=False, eval_metric='logloss')

# xgb_model.fit(X_train, y_train)

# # Calculate the mean absolute error of your Random Forest model on the validation data
# val_predictions_xgb = xgb_model.predict(X_val) 
# pred_new = (val_predictions_xgb > 0.5).astype(int)
# xgb_val_mae = mean_absolute_error(y_val, pred_new)
# print(xgb_val_mae)



pred = model.predict(X_test)
pred


pred_new = (pred> 0.5).astype(int)


output = pd.DataFrame(columns = ['row_id', 'rule_violation'])
output['row_id'] = test_data['row_id']
output['rule_violation'] =pred_new


output.head()


output.describe()


# rm -rf '/kaggle/working/submission.csv'


output.to_csv("/kaggle/working/submission.csv", index = False)




