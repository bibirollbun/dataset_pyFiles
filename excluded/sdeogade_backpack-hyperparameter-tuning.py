import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import lightgbm as lgb
import xgboost as xgb


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Enable MirroredStrategy for multi-GPU tuning
strategy = tf.distribute.MirroredStrategy()


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
trainEX_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df.head()


train_df = pd.concat([train_df, trainEX_df], ignore_index=True)


train_df.info()


train_df.isnull().sum()  # Check missing values


plt.figure(figsize=(8, 5))
sns.histplot(train_df['Price'], bins=50, kde=True, color='blue')
plt.title('Distribution of Backpack Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# Preprocessing: Handle missing values
for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])  # Use train mode for consistency

train_df['Laptop Compartment'] = train_df['Laptop Compartment'].fillna('No')
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna('No')

train_df['Waterproof'] = train_df['Waterproof'].fillna('No')
test_df['Waterproof'] = test_df['Waterproof'].fillna('No')


plt.figure(figsize=(8, 5))
sns.histplot(train_df['Weight Capacity (kg)'], bins=50, kde=True, color='purple')
plt.title('Weight Capacity Distribution')
plt.xlabel('Weight Capacity (kg)')
plt.show()


train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].median())
test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].median())


# Feature Engineering
def add_features(df):
    df['Compartments_per_Weight'] = df['Compartments'] / df['Weight Capacity (kg)'].replace(0, 1)  # Avoid division by zero
    df['Is_Premium_Brand'] = df['Brand'].isin(['Nike', 'Adidas', 'Under Armour']).astype(int)
    df['Size_Ordinal'] = df['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3}).fillna(2)  # Default to Medium
    df['Feature_Rich'] = (df['Laptop Compartment'] == 'Yes') & (df['Waterproof'] == 'Yes')
    df['Feature_Rich'] = df['Feature_Rich'].astype(int)
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)


# Verify no missing values
print("Train Missing Values:\n", train_df.isnull().sum())
print("Test Missing Values:\n", test_df.isnull().sum())


# Define features
num_features = ['Compartments', 'Weight Capacity (kg)', 'Compartments_per_Weight', 'Size_Ordinal']
categorical_cols = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof', 'Is_Premium_Brand', 'Feature_Rich']


colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink' ,'yellow',]  # Add more colors if needed

plt.figure(figsize=(15, 10))

for i, col in enumerate(categorical_cols):
    plt.subplot(2, 4, i+1)  # Adjust subplot grid as needed
    train_df[col].value_counts().plot(kind='bar', color=colors[i])
    plt.title(f'Value Counts Distribution of {col}')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout() # Adjust layout to prevent overlapping

plt.show()



# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])


# Prepare training data
X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Transform test data (without 'id')
X_test = test_df.drop(columns=['id'])
test_processed = preprocessor.transform(X_test)


# ------ LightGBM with GridSearchCV ------
lgb_model = lgb.LGBMRegressor(random_state=42)


lgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10],
    'num_leaves': [31, 50]
}


lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
lgb_grid.fit(X_train_processed, y_train)


print("Best LightGBM Params:", lgb_grid.best_params_)


lgb_pred = lgb_grid.predict(X_val_processed)
lgb_rmse = mean_squared_error(y_val, lgb_pred, squared=False)
print(f"LightGBM RMSE: {lgb_rmse}")


# ------ Enhanced Neural Network in Keras ------
def build_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(X_train_processed.shape[1],)),
        layers.Dense(hp.Int('units_1', 256, 512, 128), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float('dropout_1', 0.1, 0.3, step=0.1)),
        
        layers.Dense(hp.Int('units_2', 128, 256, 64), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1)),
        
        layers.Dense(hp.Int('units_3', 64, 128, 32), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        
        layers.Dense(hp.Int('units_4', 32, 64, 16), activation='relu', kernel_initializer='he_normal'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model


tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_rmse", direction="min"),
        max_epochs=10,
        factor=3,
        directory='hyperparameter_tuning_v2',
        project_name='backpack_price_prediction'
    )

# Hyperparameter search
tuner.search(X_train_processed, y_train, validation_data=(X_val_processed, y_val), epochs=10, batch_size=128)


# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")


# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)

with strategy.scope():
    best_model = tuner.hypermodel.build(best_hps)
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
        loss=keras.losses.Huber(delta=1.5),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = best_model.fit(X_train_processed, y_train, validation_data=(X_val_processed, y_val), epochs=5, batch_size=128, callbacks=[early_stopping,lr_schedule])


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.title('Training & Validation RMSE')
plt.tight_layout()
plt.show()


# Evaluate neural network
y_pred_nn = best_model.predict(X_val_processed)
nn_rmse = mean_squared_error(y_val, y_pred_nn, squared=False)
print(f"Neural Network RMSE: {nn_rmse}")


# ------ Submission ------
# Predictions for each model
lgb_test_pred = lgb_grid.predict(test_processed)
nn_test_pred = best_model.predict(test_processed).flatten()

# Individual submissions
submission_lgb = pd.DataFrame({'id': test_df['id'], 'Price': lgb_test_pred})
submission_lgb.to_csv('submission_lgb.csv', index=False)
print(f"Submission file created for LightGBM: submission_lgb.csv (RMSE: {lgb_rmse})")

submission_nn = pd.DataFrame({'id': test_df['id'], 'Price': nn_test_pred})
submission_nn.to_csv('submission_nn.csv', index=False)
print(f"Submission file created for Neural Network: submission_nn.csv (RMSE: {nn_rmse})")


# Ensemble: Simple averaging of all three models
ensemble_test_pred = (lgb_test_pred + nn_test_pred) / 3
submission_ensemble = pd.DataFrame({'id': test_df['id'], 'Price': ensemble_test_pred})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

# Evaluate ensemble RMSE on validation set for reference
ensemble_val_pred = (lgb_pred + y_pred_nn) / 3
ensemble_rmse = mean_squared_error(y_val, ensemble_val_pred, squared=False)
print(f"Submission file created for Ensemble (Average): submission_ensemble.csv (Validation RMSE: {ensemble_rmse})")


# Choose the best individual model based on RMSE (for reference)
best_rmse = min(lgb_rmse, xgb_rmse, nn_rmse)
if best_rmse == lgb_rmse:
    best_model_name = "LightGBM"
else:
    best_model_name = "Neural Network"
print(f"Best individual model based on validation RMSE: {best_model_name} (RMSE: {best_rmse})")




