import numpy as np
import pandas as pd
import os

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


print("Import Completed!!")


sample_submission_filePath  =  "/kaggle/input/playground-series-s5e3/sample_submission.csv"
train_filePath = "/kaggle/input/playground-series-s5e3/train.csv"
test_filePath = "/kaggle/input/playground-series-s5e3/test.csv"

# Reading .csv file
train_data = pd.read_csv(train_filePath)
test_data = pd.read_csv(test_filePath)
sample_submission = pd.read_csv(sample_submission_filePath)
print("Read Completed!!!")


def missing_details(df):
    print("\n------------------------------------------------")
    print("Missing Values :")
    print("------------------------------------------------")
    print( df.isnull().sum()[df.isnull().sum() > 0] )
    
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100 
    print("\n------------------------------------------------")
    print("Percentage of Missing values: (%) ")
    print("------------------------------------------------")
    print(missing_percentage[missing_percentage > 0])
    

    
    total_missing_percentage = (df.isnull().sum().sum() / (df.size)) * 100
    print("\n------------------------------------------------")
    print(f"Total missing values percentage: {total_missing_percentage:.2f}%")
    print("------------------------------------------------")


print("\n****************************************************")
print("Missing Details of train_data: ")
print("\n****************************************************")
missing_details(train_data)

print("\n****************************************************")
print("Missing Details of test_data: ")
print("\n****************************************************")
missing_details(test_data)


def imputation(train, test):
    # my_imputer = KNNImputer(n_neighbors=10)
    my_imputer = SimpleImputer(strategy='median')
    train_drop = train.drop(columns=['rainfall'])
    train_imputed = pd.DataFrame(my_imputer.fit_transform(train_drop))
    test_imputed = pd.DataFrame(my_imputer.transform(test))

    test_imputed.columns = test.columns
    
    return test_imputed


new_test_data = imputation(train_data, test_data)
missing_details(new_test_data)


print(train_data.shape)

X = train_data.drop(['rainfall'], axis=1)
y = train_data.rainfall

print(X.shape, y.shape)

train_size = int(0.8* len(train_data))

X_train = X.iloc[:train_size]
X_valid = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_valid = y.iloc[train_size:]
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


early_stopping = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.0001, 
    patience=20, 
    mode="min", 
    restore_best_weights=True
)



model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu',  kernel_initializer='he_normal',  input_shape=[13]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu',  kernel_initializer='he_normal'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu',  kernel_initializer='he_normal'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),
])
print("Model created!")


optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=32,
    epochs=2000,
    callbacks=[early_stopping],
)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot();


y_pred = model.predict(X_valid)
y_pred_prob = model.predict(X_valid)  # Get probabilities (0 to 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert to 0 or 1

print(classification_report(y_valid, y_pred))  # Shows precision, recall, F1-score



final_prediction = model.predict(new_test_data)
final_prediction = final_prediction.flatten() 
submission = pd.DataFrame({'id': new_test_data['id'], 'rainfall': final_prediction})
submission.to_csv('submission.csv', index=False)

