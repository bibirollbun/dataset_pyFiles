import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")                    


train.head()


train.info()


train.describe()


test.head()


test.info()


test.describe()


train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
train = train.drop('id', axis=1)

train = train.groupby(['Sex','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].median().reset_index()


train.head()


train.info()


# For training set
train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
train['Intensity'] = train['Heart_Rate'] / train['Duration']
train['Severity'] = train['Body_Temp'] / train['Duration']
train['Heart_Age'] = train['Heart_Rate'] / train['Age']
train['Workload'] = train['Weight'] / train['Duration']
train['Heart_Weight'] = train['Heart_Rate'] / train['Weight']
train['Heart_BMI'] = train['Heart_Rate'] / train['BMI']
train['Duration_BMI'] = train['Duration'] / train['BMI']
train['Strain'] = train['Intensity'] + train['Severity'] + train['Workload']
train['Temp_Rise'] = train['Body_Temp'] - 37   # normal body temp
train['Temp_Rate'] = train['Temp_Rise'] / train['Duration']
train['Heart_Rise'] = train['Heart_Rate'] - 60  # resting heart rate
train['Heart_Soar'] = train['Heart_Rise'] / train['Duration']
train['Body_Fat'] = (1.2 * train['BMI']) + (0.23 * train['Age']) - (10.8 * train['Sex']) - 5.4 # Deurenberg formula 
train['Fat_Mass'] = (train['Body_Fat'] / 100) * train['Weight']   # actual fat mass using percentage
train['Fat_Free_Mass'] = train['Weight'] - train['Fat_Mass']
train['LBM_M'] = (0.407 * train['Weight']) + (0.267 * train['Height']) - 19.2  # Lean body mass for men
train['LBM_F'] = (0.252 * train['Weight']) + (0.473 * train['Height']) - 48.3  # Lean body mass for women
train['LBM'] = np.where(train['Sex'] == 1, train['LBM_M'], train['LBM_F'])     # Combine for both sexes
train.drop(columns=['LBM_M', 'LBM_F'], inplace=True)

# For test set
test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2
test['Intensity'] = test['Heart_Rate'] / test['Duration']
test['Severity'] = test['Body_Temp'] / test['Duration']
test['Heart_Age'] = test['Heart_Rate'] / test['Age']
test['Workload'] = test['Weight'] / test['Duration']
test['Heart_Weight'] = test['Heart_Rate'] / test['Weight']
test['Heart_BMI'] = test['Heart_Rate'] / test['BMI']
test['Duration_BMI'] = test['Duration'] / test['BMI']
test['Strain'] = test['Intensity'] + test['Severity'] + test['Workload']
test['Temp_Rise'] = test['Body_Temp'] - 37   # normal body temp
test['Temp_Rate'] = test['Temp_Rise'] / test['Duration']
test['Heart_Rise'] = test['Heart_Rate'] - 60  # resting heart rate
test['Heart_Soar'] = test['Heart_Rise'] / test['Duration']
test['Body_Fat'] = (1.2 * test['BMI']) + (0.23 * test['Age']) - (10.8 * test['Sex']) - 5.4 # Deurenberg formula 
test['Fat_Mass'] = (test['Body_Fat'] / 100) * test['Weight']   # actual fat mass using percentage
test['Fat_Free_Mass'] = test['Weight'] - test['Fat_Mass']
test['LBM_M'] = (0.407 * test['Weight']) + (0.267 * test['Height']) - 19.2  # Lean body mass for men
test['LBM_F'] = (0.252 * test['Weight']) + (0.473 * test['Height']) - 48.3  # Lean body mass for women
test['LBM'] = np.where(test['Sex'] == 1, test['LBM_M'], test['LBM_F'])      # Combine for both sexes
test.drop(columns=['LBM_M', 'LBM_F'], inplace=True)



train.head()


test.head()


train.describe()


# Compute correlation matrix
corr_matrix = train.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Feature Correlation Heatmap (Including Calories)")
plt.tight_layout()
plt.show()


def add_cross_terms(df, features):
    cross_terms = {}  # use a dictionary to collect new columns
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            col_name = f"{features[i]}_x_{features[j]}"
            cross_terms[col_name] = df[features[i]] * df[features[j]]
    
    # Convert to DataFrame and concat once
    cross_df = pd.DataFrame(cross_terms)
    return pd.concat([df, cross_df], axis=1)


numerical = ["Age", "Height", "Weight", "Duration", 
             "Heart_Rate", "Body_Temp", "BMI", "Intensity", 
             "Severity", "Heart_Age", "Workload", 
             "Heart_Weight", "Heart_BMI", "Duration_BMI", 
             "Strain", "Temp_Rise", "Temp_Rate", 
             "Heart_Rise", "Heart_Soar", "Body_Fat",
             "Fat_Mass", "Fat_Free_Mass", "LBM"
            ]

train = add_cross_terms(train, numerical)
test = add_cross_terms(test, numerical)


train.head()


test.head()


X = train.drop(columns=['Calories', 'Sex'])
y = np.log1p(train['Calories'])

X.head()


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_final = test.drop(columns=['id', 'Sex'])
test_scaled = scaler.fit_transform(test_final)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.head()


X_scaled_df['Calories'] = y.values  # Add target back for correlation

# Compute correlation matrix
corr_matrix2 = X_scaled_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix2, annot=False, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Feature Correlation Heatmap, Extended")
plt.tight_layout()
plt.show()


FOLDS = 7

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(X_scaled))
pred = np.zeros(len(test_scaled))

X_train_df = pd.DataFrame(X_scaled, columns=X.columns)

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\n{'#'*3} Fold {i+1} {'#'*3}")
    
    x_train = X_train_df.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X_train_df.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = test_scaled.copy()
    
    start = time.time()

    model = Sequential([
                Input(shape=(x_train.shape[1],)),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)  # Linear activation for regression
            ])

    # Learning rate reducer
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',     # Watch validation loss
        factor=0.8,             # (1-n) reduction on the learning rate
        patience=1,             # Wait for n epochs of no improvement
        verbose=0,              # Print updates
        min_lr=1e-7             # Don’t go below this
    )
    
    # Prevent overfitting 
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Combine
    callbacks = [lr_reducer, early_stop, checkpoint]
    
    def rmsle(y_true, y_pred):
        y_true_exp = tf.math.expm1(y_true)
        y_pred_exp = tf.math.expm1(y_pred)
        return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred_exp) - tf.math.log1p(y_true_exp))))
    
    optimizer = Adam(learning_rate=0.002)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=[rmsle])
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=50,
        batch_size=256,
        callbacks=callbacks, 
        verbose=0
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid).ravel()
    pred += model.predict(x_test).ravel()

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Training time: {time.time() - start:.1f} sec")

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Curve')
    plt.ylim(top=0.01, bottom=0.0034)
    plt.show()

# Average test predictions
pred /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.5f}")


# Convert back to original scale
y_preds = np.expm1(pred)
print('predict mean :',y_preds.mean())
print('predict median :',np.median(y_preds))

y_preds = np.clip(y_preds,1,314)
print('predict mean after clip:',y_preds.mean())
print('predict median after clip:',np.median(y_preds))

submission = pd.DataFrame({
    'id': test['id'],        # or another identifier column if available
    'Calories': y_preds.flatten()
})

submission.to_csv('submission.csv', index=False)

print("Done!")

submission.head()

