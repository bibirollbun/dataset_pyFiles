import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
DATA_DIR = '/kaggle/input/playground-series-s5e10'
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "accident_risk"
ID_COL = "id"
N_FOLDS = 5
EPOCHS_AE = 50
EPOCHS_REG = 80
BATCH_SIZE = 512

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ---------------- FEATURE TYPES ----------------
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c != TARGET_COL and c != ID_COL]

# ---------------- ENCODE CATEGORICALS ----------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# ---------------- SCALE NUMERICALS ----------------
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# ---------------- CREATE AUTOENCODER INPUT ----------------
X = train_df[categorical_cols + numerical_cols].values
y = train_df[TARGET_COL].values
X_test = test_df[categorical_cols + numerical_cols].values

# ---------------- CUSTOM RMSE LOSS ----------------
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------- AUTOENCODER MODEL ----------------
embedding_sizes = {col: min(50, (train_df[col].nunique() + 1)//2) for col in categorical_cols}

inputs = []
embeddings = []

# Categorical embeddings
for col in categorical_cols:
    input_cat = tf.keras.Input(shape=(1,), name=col)
    embed_dim = embedding_sizes[col]
    embed = tf.keras.layers.Embedding(input_dim=train_df[col].nunique(), output_dim=embed_dim)(input_cat)
    embed = tf.keras.layers.Flatten()(embed)
    inputs.append(input_cat)
    embeddings.append(embed)

# Numerical input
num_input = tf.keras.Input(shape=(len(numerical_cols),), name='numerical')
inputs.append(num_input)
embeddings.append(num_input)

# Concatenate all
x = tf.keras.layers.Concatenate()(embeddings)

# Encoder
x = tf.keras.layers.Dense(256, activation='gelu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='gelu')(x)
latent = tf.keras.layers.Dense(256, activation='gelu', name='latent_space')(x)  # Increased latent dim

# Decoder
x = tf.keras.layers.Dense(128, activation='gelu')(latent)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='gelu')(x)
output = tf.keras.layers.Dense(X.shape[1], activation='linear', name='reconstruction')(x)

autoencoder = tf.keras.Model(inputs=inputs, outputs=output)
autoencoder.compile(optimizer='adam', loss=rmse)

# ---------------- INPUT DICTIONARY ----------------
def dict_input(X_array):
    data_dict = {}
    for i, col in enumerate(categorical_cols):
        data_dict[col] = X_array[:, i]
    num_start = len(categorical_cols)
    data_dict["numerical"] = X_array[:, num_start:]
    return data_dict

train_dict = dict_input(X)
X_test_dict = dict_input(X_test)

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# ---------------- TRAIN AUTOENCODER ----------------
autoencoder.fit(
    train_dict, X,
    validation_split=0.1,
    epochs=EPOCHS_AE,
    batch_size=BATCH_SIZE,
    verbose=2,
    callbacks=[early_stop, reduce_lr]
)

# ---------------- EXTRACT LATENT FEATURES ----------------
encoder = tf.keras.Model(inputs=inputs, outputs=autoencoder.get_layer('latent_space').output)
latent_features = encoder.predict(train_dict)
latent_test_features = encoder.predict(X_test_dict)

# ---------------- 5-FOLD REGRESSOR ----------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros(X_test.shape[0])
fold_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(latent_features)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_tr, X_val = latent_features[train_idx], latent_features[val_idx]
    y_tr, y_val_fold = y[train_idx], y[val_idx]
    
    latent_input = tf.keras.Input(shape=(latent_features.shape[1],))
    x = tf.keras.layers.Dense(128, activation='gelu')(latent_input)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='gelu')(x)
    x = tf.keras.layers.Dense(32, activation='gelu')(x)
    output_pred = tf.keras.layers.Dense(1, activation='linear')(x)
    
    regressor = tf.keras.Model(latent_input, output_pred)
    regressor.compile(optimizer='adam', loss=rmse)
    
    regressor.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val_fold),
        epochs=EPOCHS_REG,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )
    
    val_pred = regressor.predict(X_val).flatten()
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    fold_rmse_scores.append(val_rmse)
    print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")
    
    test_preds += regressor.predict(latent_test_features).flatten() / N_FOLDS

print("\nAverage RMSE across folds:", np.mean(fold_rmse_scores))

# ---------------- CREATE KAGGLE SUBMISSION ----------------
submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    TARGET_COL: test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")





