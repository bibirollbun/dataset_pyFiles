import numpy as np
import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Separate features
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
categorical_features = ['Sex']

# Numerical transformer
numerical_transformer = Pipeline([
    ('minmax', MinMaxScaler())
])

# Categorical transformer using OneHotEncoder
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)],
    
        remainder='passthrough', # Keep other columns that are not transformer
        verbose_feature_names_out=False # Set to False to avoid prefixing transformer names
)
preprocessor.set_output(transform="pandas")


train_df = train.drop('Calories', axis=1)
y = train['Calories']


preprocessor.fit(train_df)
X = preprocessor.transform(train_df)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

tf.random.set_seed(42)

model = tf.keras.Sequential([    
    # First block
    tf.keras.layers.Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    # Second block
    tf.keras.layers.Dense(192, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    # Third block
    tf.keras.layers.Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Fourth block
    tf.keras.layers.Dense(96, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Fifth block
    tf.keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    
    # Sixth block
    tf.keras.layers.Dense(48, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])



model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse']) 


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_loss',        
    patience=15,               
    restore_best_weights=True, 
    min_delta=0.001,           
    mode='min',                
    baseline=None,             
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',        
    factor=0.2,                
    patience=3,                
    min_lr=0.000001,           
    min_delta=0.001,           
    cooldown=1,                
    verbose=1
)


history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=1)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7)) 

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Training History') 
plt.xlabel('Epoch') 
plt.ylabel('Metric Value')
plt.legend() 
plt.grid(True) 
plt.style.use('seaborn-v0_8-darkgrid')

plt.show()





from sklearn.metrics import mean_squared_log_error

y_train_pred = model.predict(X_train)
y_train_pred_clipped = np.maximum(0, y_train_pred.flatten())
y_train_clipped = np.maximum(0, y_train)

rmsle_train = np.sqrt(mean_squared_log_error(y_train_clipped, y_train_pred_clipped))
print(f"Training RMSLE: {rmsle_train}")

y_val_pred = model.predict(X_test)
y_val_pred_clipped = np.maximum(0, y_val_pred.flatten())
y_test_clipped = np.maximum(0, y_test)

rmsle_val = np.sqrt(mean_squared_log_error(y_test_clipped, y_val_pred_clipped))
print(f"Validation RMSLE: {rmsle_val}")


#input_shape = train_processed.shape[1]
#odel.build(input_shape)
#keras.utils.plot_model(model, show_shapes=True)


#preprocessor.fit(test)
test_transformed = preprocessor.transform(test)
predictions = model.predict(test_transformed)


#predictions_clipped = np.maximum(0, predictions.flatten())
# %%
submission = pd.DataFrame({'id': test.id, 'Calories': predictions.flatten()}) 
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(submission.head())

