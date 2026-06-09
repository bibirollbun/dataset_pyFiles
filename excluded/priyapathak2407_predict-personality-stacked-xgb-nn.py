import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler # For normalizing
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_data = train_data.drop_duplicates()


train_data.head()


train_nan_columns = train_data.columns[train_data.isna().any()].tolist()
train_nan_columns


train_data['Time_spent_Alone'] = train_data['Time_spent_Alone'].fillna(train_data['Time_spent_Alone'].median())
train_data['Stage_fear'] = train_data['Stage_fear'].fillna(train_data['Stage_fear'].mode())
train_data['Social_event_attendance'] = train_data['Social_event_attendance'].fillna(train_data['Social_event_attendance'].mean())
train_data['Going_outside'] = train_data['Going_outside'].fillna(train_data['Going_outside'].mean())
train_data['Drained_after_socializing'] = train_data['Drained_after_socializing'].fillna(train_data['Drained_after_socializing'].mode())
train_data['Friends_circle_size'] = train_data['Friends_circle_size'].fillna(train_data['Friends_circle_size'].median())
train_data['Post_frequency'] = train_data['Post_frequency'].fillna(train_data['Post_frequency'].median())


test_nan_columns = test_data.columns[test_data.isna().any()].tolist()
test_nan_columns


test_data['Time_spent_Alone'] = test_data['Time_spent_Alone'].fillna(test_data['Time_spent_Alone'].median())
test_data['Stage_fear'] = test_data['Stage_fear'].fillna(test_data['Stage_fear'].mode())
test_data['Social_event_attendance'] = test_data['Social_event_attendance'].fillna(test_data['Social_event_attendance'].mean())
test_data['Going_outside'] = test_data['Going_outside'].fillna(test_data['Going_outside'].mean())
test_data['Drained_after_socializing'] = test_data['Drained_after_socializing'].fillna(test_data['Drained_after_socializing'].mode())
test_data['Friends_circle_size'] = test_data['Friends_circle_size'].fillna(test_data['Friends_circle_size'].median())
test_data['Post_frequency'] = test_data['Post_frequency'].fillna(test_data['Post_frequency'].median())


x = train_data.drop('Personality', axis=1)
y = train_data['Personality']


# Initialize encoder
encoder = LabelEncoder()

# Convert labels to numbers
y = encoder.fit_transform(y)


# sanity check
print(x.columns)
print(len(x)==len(y))
print(len(x.columns))


# Splitting the train data into train and validation
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.005, random_state=44)


print('Length of training data: ',len(train_data))
print('Length of testing data: ',len(test_data))


print('Length of train data: ',len(train_x))
print('Length of val data: ', len(val_x))


columns = train_x.columns


columns


train_x.describe()


# converting the Stage_fear column to one hot-encoding
train_x = pd.get_dummies(train_x, columns=['Stage_fear'])
val_x = pd.get_dummies(val_x, columns=['Stage_fear'])
test_data = pd.get_dummies(test_data, columns=['Stage_fear'])


# converting the Drained_after_socializing column to one hot-encoding
train_x = pd.get_dummies(train_x, columns=['Drained_after_socializing'])
val_x = pd.get_dummies(val_x, columns=['Drained_after_socializing'])
test_data = pd.get_dummies(test_data, columns=['Drained_after_socializing'])


# dropping id as it is just sequence and not contributing anything to result
train_x = train_x.drop('id', axis=1)
val_x = val_x.drop('id', axis=1)
test_data = test_data.drop('id', axis=1)


train_x = train_x.astype(np.float32)
val_x = val_x.astype(np.float32)
train_y = train_y.astype(np.int32)
val_y = val_y.astype(np.int32)


input_dim = len(train_x.columns)
n_classes = 2


# List of seed values
seeds = [1, 32, 67,100]
base_models = []
n_base = len(seeds)+3 # Adding one for 2 nn to be stacked
pred_train = np.zeros((train_x.shape[0], n_base))
pred_test = np.zeros((test_data.shape[0], n_base))

for i, seed in enumerate(seeds):
    model = XGBClassifier(objective='binary:logistic', random_state=seed, n_estimators=200)
    model.fit(train_x, train_y)
    base_models.append(model)
    pred_train[:, i] = model.predict(train_x)
    pred_test[:, i] = model.predict(test_data)


# Train Neural Network
nn_model_1 = Sequential([
    Dense(32, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model_1.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
nn_model_1.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)

# Neural net predictions (round for classification)
pred_train[:, -1] = nn_model_1.predict(train_x).flatten() > 0.5
pred_test[:, -1] = nn_model_1.predict(test_data).flatten() > 0.5


# Train Neural Network
nn_model_2 = Sequential([
    Dense(200, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model_2.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
nn_model_2.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)

# Neural net predictions (round for classification)
pred_train[:, -2] = nn_model_2.predict(train_x).flatten() > 0.5
pred_test[:, -2] = nn_model_2.predict(test_data).flatten() > 0.5


nn_model_3 = Sequential([
    Dense(400, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(300, activation='relu'),
    Dense(100, activation='relu'),
    Dense(20, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model_3.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
nn_model_3.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)

# Neural net predictions (round for classification)
pred_train[:, -3] = nn_model_3.predict(train_x).flatten() > 0.5
pred_test[:, -3] = nn_model_3.predict(test_data).flatten() > 0.5


# Train the meta-model
meta_model = LogisticRegression()
meta_model.fit(pred_train, train_y)

# Obtain stacked predictions
meta_train_pred = meta_model.predict(pred_train)
meta_test_pred = meta_model.predict(pred_test)


meta_test_pred[1]


# For going from one hot encoding to labels
y_res = encoder.inverse_transform(meta_test_pred)


# Sanity check for result and test_data
len(test_data)==len(y_res)
print('Len test data: ',len(test_data))
print('Len y_res: ',len(y_res))


# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission["Personality"] = y_res
submission.to_csv("submission.csv", index=False)
print('Sumission done!')
submission.head()




