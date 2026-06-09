# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files_names = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files_names.append(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#print(files_names)
sample_s = pd.read_csv(files_names[0])
train_data = pd.read_csv(files_names[1])
test_data = pd.read_csv(files_names[2])

y_train = train_data['Depression']
ids = test_data['id']
train_data.drop(['Depression'], axis=1, inplace=True)
train_data['Study/work_Pressure'] = train_data[['Academic Pressure', 'Work Pressure']].mean(axis=1)
train_data['Study/Job Satisfaction'] = train_data[['Study Satisfaction', 'Job Satisfaction']].mean(axis=1)
train_data.drop(['id', 'Name', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

test_data['Study/work_Pressure'] = test_data[['Academic Pressure', 'Work Pressure']].mean(axis=1)
test_data['Study/Job Satisfaction'] = test_data[['Study Satisfaction', 'Job Satisfaction']].mean(axis=1)
test_data.drop(['id', 'Name', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

#print(train_data.head(10))

# Categorical
train_data['Profession'] = train_data['Profession'].fillna('Unknown')
test_data['Profession'] = test_data['Profession'].fillna('Unknown')

train_data['Dietary Habits'] = train_data['Dietary Habits'].fillna(train_data['Dietary Habits'].mode()[0])
test_data['Dietary Habits'] = test_data['Dietary Habits'].fillna(test_data['Dietary Habits'].mode()[0])

train_data['Degree'] = train_data['Degree'].fillna(train_data['Degree'].mode()[0])
test_data['Degree'] = test_data['Degree'].fillna(test_data['Degree'].mode()[0])



# Numerical
train_data['CGPA'] = train_data['CGPA'].fillna(train_data['CGPA'].mean())
test_data['CGPA'] = test_data['CGPA'].fillna(test_data['CGPA'].mean())

train_data['Low_CGPA'] = (train_data['CGPA'] < 6.5).astype(int)
test_data['Low_CGPA'] = (test_data['CGPA'] < 6.5).astype(int)

train_data['Study/work_Pressure'] = train_data['Study/work_Pressure'].fillna(train_data['Study/work_Pressure'].mean())
test_data['Study/work_Pressure'] = test_data['Study/work_Pressure'].fillna(test_data['Study/work_Pressure'].mean())

train_data['Study/Job Satisfaction'] = train_data['Study/Job Satisfaction'].fillna(train_data['Study/Job Satisfaction'].mean())
test_data['Study/Job Satisfaction'] = test_data['Study/Job Satisfaction'].fillna(test_data['Study/Job Satisfaction'].mean())

binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0, 'Student': 0, 'Working Professional': 1}
train_data['Gender'] = train_data['Gender'].map(binary_map)
train_data['Working Professional or Student'] = train_data['Working Professional or Student'].map(binary_map)
train_data['Have you ever had suicidal thoughts ?'] = train_data['Have you ever had suicidal thoughts ?'].map(binary_map)
train_data['Family History of Mental Illness'] = train_data['Family History of Mental Illness'].map(binary_map)

test_data['Gender'] = test_data['Gender'].map(binary_map)
test_data['Working Professional or Student'] = test_data['Working Professional or Student'].map(binary_map)
test_data['Have you ever had suicidal thoughts ?'] = test_data['Have you ever had suicidal thoughts ?'].map(binary_map)
test_data['Family History of Mental Illness'] = test_data['Family History of Mental Illness'].map(binary_map)



# Sleep Duration
sleep_map = {
    'Less than 5 hours': 4.5,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9.0
}
train_data['Sleep Duration'] = train_data['Sleep Duration'].map(sleep_map)
test_data['Sleep Duration'] = test_data['Sleep Duration'].map(sleep_map)
train_data['Low_Sleep'] = (train_data['Sleep Duration'] < 5).astype(int)
test_data['Low_Sleep'] = (test_data['Sleep Duration'] < 5).astype(int)
train_data['Sleep Duration'] = train_data['Sleep Duration'].fillna(train_data['Sleep Duration'].mean())
test_data['Sleep Duration'] = test_data['Sleep Duration'].fillna(test_data['Sleep Duration'].mean())
train_data['High_Stress'] = (train_data['Financial Stress'] > 3).astype(int)
test_data['High_Stress'] = (test_data['Financial Stress'] > 3).astype(int)
train_data['Financial Stress'] = train_data['Financial Stress'].fillna(train_data['Financial Stress'].mean())
test_data['Financial Stress'] = test_data['Financial Stress'].fillna(test_data['Financial Stress'].mean())
# Dietary Habits
diet_map = {
    'Unhealthy': 0,
    'Moderate': 1,
    'Healthy': 2
}
train_data['Dietary Habits'] = train_data['Dietary Habits'].map(diet_map)
test_data['Dietary Habits'] = test_data['Dietary Habits'].map(diet_map)
train_data['Dietary Habits'] = train_data['Dietary Habits'].fillna(train_data['Dietary Habits'].mean())
test_data['Dietary Habits'] = test_data['Dietary Habits'].fillna(test_data['Dietary Habits'].mean())
#train_data = train_data.fillna(train_data.mean(), inplace=True)

#train_data = pd.get_dummies(train_data, columns=['City', 'Profession', 'Degree'], drop_first=True)
#test_data = pd.get_dummies(test_data, columns=['City', 'Profession', 'Degree'], drop_first=True)
train_data.drop(['City', 'Profession', 'Degree'], axis=1, inplace=True)
test_data.drop(['City', 'Profession', 'Degree'], axis=1, inplace=True)

nan_cols = train_data.isna().sum()
nan_cols = nan_cols[nan_cols > 0]
print(nan_cols)

print("NaNs in train data:", train_data.isnull().sum().sum())
print("Infs in train data:", np.isinf(train_data.values).sum())

print("NaNs in y_train:", np.isnan(y_train).sum())
print("Infs in y_train:", np.isinf(y_train).sum())

#print(test_data)


from sklearn.preprocessing import StandardScaler

# X_train: your feature matrix from training data
# y_train: target column

numeric_cols_train = train_data.select_dtypes(include=['int64', 'float64']).columns
numeric_cols_test = test_data.select_dtypes(include=['int64', 'float64']).columns
# Assuming you already dropped/encoded categorical columns
# Fit scaler on training data only
scaler = StandardScaler()
train_data[numeric_cols_train] = scaler.fit_transform(train_data[numeric_cols_train])

# Apply same transformation to test data
test_data[numeric_cols_test] = scaler.fit_transform(test_data[numeric_cols_test])
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("Train head:", train_data.head())
print("Test head:", test_data.head())



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
import numpy as np

# Make sure data is numeric
#train_data[numeric_cols_train] = np.asarray(train_data[numeric_cols_train]).astype('float32')
#y_train = np.asarray(y_train).astype('float32')

# Define model
model = Sequential([
    Input(shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-5, verbose=1)

history = model.fit(train_data, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Over Epochs')
plt.show()


# Predict probabilities
predictions = model.predict(test_data)


predictions_cleaned = np.nan_to_num(predictions, nan=0.0)

predictions_binary = (predictions_cleaned > 0.5).astype(int).flatten()

submission = pd.DataFrame({
    "id": ids,
    "Depression": predictions_binary  # sicherheitshalber als int (0/1)
})

# Als CSV speichern
submission.to_csv("submission.csv", index=False)

# Convert to binary labels
#y_pred = (y_pred_probs > 0.5).astype(int).flatten()

