import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input




sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train =  pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra =pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


print("Train sütunları:", train.columns.tolist())
print("Test sütunları:", test.columns.tolist())


train = pd.concat([train, train_extra], axis=0, ignore_index=True)
print("Birleştirilmiş train boyutu:", train.shape)


possible_categorical_cols = ["Brand", "Size", "Laptop Compartment", "Waterproof", "Style", "Color", "Material"]
possible_numeric_cols = ["Compartments", "Weight Capacity (kg)"]


categorical_cols = [col for col in possible_categorical_cols if col in train.columns]
numeric_cols = [col for col in possible_numeric_cols if col in train.columns]


for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])
for col in numeric_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())


if "Material" in train.columns:
    train = pd.get_dummies(train, columns=["Material"], prefix="Material")
    test = pd.get_dummies(test, columns=["Material"], prefix="Material")
    train, test = train.align(test, join="left", axis=1, fill_value=0)



print("Özellik mühendisliği başlıyor...")
# Yeni özellik: Functional Score (eğer gerekli sütunlar varsa)
if "Weight Capacity (kg)" in train.columns and "Compartments" in train.columns:
    train["Functional Score"] = train["Weight Capacity (kg)"] * train["Compartments"]
    test["Functional Score"] = test["Weight Capacity (kg)"] * test["Compartments"]
    numeric_cols.append("Functional Score")


size_map = {"Small": 1, "Medium": 2, "Large": 3}
if "Size" in train.columns:
    train["Size Numeric"] = train["Size"].map(size_map)
    test["Size Numeric"] = test["Size"].map(size_map)
    numeric_cols.append("Size Numeric")



le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le



scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])


X = train.drop(columns=["id", "Price"])
y = train["Price"].values
X_test = test.drop(columns=["id"])



X = X.fillna(0)
X_test = X_test.fillna(0)



def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))  # Input katmanı
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # Regresyon çıkışı
    model.compile(optimizer="adam", loss="mse")
    return model


print("Model eğitimi başlıyor...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))



for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Model oluşturma
    model = create_model(input_shape=X_train.shape[1])
    
    # Erken durdurma
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    
    # Model eğitimi
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Validation tahmini
    oof_preds[val_idx] = model.predict(X_val, verbose=0).flatten()
    # Test tahmini
    test_preds += model.predict(X_test, verbose=0).flatten() / kf.n_splits
    
    # Fold RMSE
    fold_rmse = mean_squared_error(y_val, oof_preds[val_idx], squared=False)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")



cv_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"Cross-Validation RMSE: {cv_rmse:.4f}")


print("Submission dosyası hazırlanıyor...")
submission = sample_submission.copy()
submission["Price"] = test_preds
submission.to_csv("submission.csv", index=False)
print("Submission dosyası oluşturuldu: submission.csv")

