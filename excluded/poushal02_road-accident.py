import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train.head()


test.head()


train.drop('id', axis=1, inplace=True)


test_id = test['id']
test.drop('id', inplace=True, axis=1)


train.shape


train.info()


train.describe()


nums = []
cats = []

for each in train.columns:
    if train[each].dtype == 'object' or train[each].dtype == 'bool':
        cats.append(each)
    else:
        nums.append(each)

print(f"Numericals {nums}\nCategoricals {cats}")


X = train.drop('accident_risk', axis=1)
y = train['accident_risk']


X


nums.pop()


nums


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

ole = OrdinalEncoder()
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('cat', ole, cats),
    ('num', scaler, ['speed_limit'])
], remainder='passthrough')

pipe = Pipeline([
    ('preprocessing', preprocessor)
])
pipe.set_output(transform="pandas")
X_processed = pipe.fit_transform(X)


X_processed.head(10)


X_processed['cat__school_season'].unique()


train['school_season'].unique()


processed_cats = []
processed_nums = []

for each in X_processed.columns:
    if 'cat' in each:
        processed_cats.append(each)
    elif 'num' in each or 'remainder' in each:
        processed_nums.append(each)


processed_cats


processed_nums


from tensorflow.keras.callbacks import EarlyStopping

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model = Sequential()

# input layer - 12 inputs
model.add(Dense(384, input_shape=(12,), activation='relu'))
model.add(Dense(192, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))

# output
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, random_state=42, test_size=0.3)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


import time

print("Time starts now")
start = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=earlystop
)

print(f"Ends after {time.time() - start}s")


test_processed = pipe.fit_transform(test)


preds = model.predict(test_processed)


test_processed.shape[0]


pred_list = []
for i in range(172585):
    pred_list.append(preds[i][0])
    


y_pred = model.predict(X_test)


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


ids = list(test_id)


submission = {
    'id' : ids,
    'accident_risk': pred_list
}


my_submission = pd.DataFrame(submission)


my_submission.to_csv("submission.csv", index=False)


model.save('road_prediction.keras')


import joblib
joblib.dump(pipe, 'pipeline.pkl')




