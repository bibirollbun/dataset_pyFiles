import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_data.head()


test_data.head()


train_data = train_data.rename(columns={'temparature': 'temperature'})
test_data = test_data.rename(columns={'temparature': 'temperature'})


def create_features(df):
    df['humidity_temp_ratio'] = df['humidity'] / (df['temperature'] + 1e-5)
    df['dewpoint_diff'] = df['dewpoint'] - df['mintemp']
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    return df


train_data = create_features(train_data)
test_data = create_features(test_data)


X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
X_test = test_data.drop(['id'], axis=1)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Imputer: fill NaNs with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_imputed)
X_test_poly = poly.transform(X_test_imputed)


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_poly, y)


X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_poly)


def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='leaky_relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='leaky_relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='leaky_relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

model = build_model(X_train_scaled.shape[1])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
test_preds = []


for train_idx, val_idx in kfold.split(X_res, y_res):
    
    X_train_fold, X_val_fold = X_res[train_idx], X_res[val_idx]
    y_train_fold, y_val_fold = y_res.iloc[train_idx], y_res.iloc[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train_fold,
        epochs=100,
        batch_size=64,
        validation_data=(X_val_scaled, y_val_fold),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    val_pred = model.predict(X_val_scaled)
    predictions.append(val_pred)
    
    test_pred = model.predict(X_test_scaled)
    test_preds.append(test_pred)


final_val_pred = np.mean(predictions, axis=0)
final_test_pred = np.mean(test_preds, axis=0)


thresholds = np.linspace(0.3, 0.7, 50)
best_threshold = 0.5
best_f1 = 0


for th in thresholds:
    current_f1 = f1_score(y_val, (final_val_pred > th).astype(int))
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = th


submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': (final_test_pred > best_threshold).astype(int).squeeze()
})

submission.to_csv('submission.csv', index=False)

print(f"Optimal Threshold: {best_threshold:.4f}")
print("Submission file created!")

