import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)
tf.config.list_physical_devices("GPU")


!find /kaggle/input -type f


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
train_df


cols = [ c for c in train_df.columns if c not in ['id', 'BeatsPerMinute'] ]
for c in cols:
    train_df[c + '_squared'] = train_df[c] ** 2
    train_df[c + '_sqrt']    = np.sqrt(np.abs(train_df[c]))  # abs() avoids NaN for negatives

for c in cols:
    test_df[c + '_squared'] = test_df[c] ** 2
    test_df[c + '_sqrt']    = np.sqrt(np.abs(test_df[c]))

X_train = train_df.drop(columns=["id","BeatsPerMinute"])
y_train = train_df["BeatsPerMinute"]
X_test  = test_df.drop(columns=["id"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


model = Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    Dense(units=256, activation='relu'),
    Dropout(.3),
    Dense(units=128, activation='relu'),
    Dropout(.2),
    Dense(units=64,  activation='relu'),
    Dropout(.1),
    Dense(units=1,   activation='linear'),
])
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]  # Mean Absolute Error for interpretability
)
model.summary()


early_stop = EarlyStopping(
    monitor="val_loss",        # you can also use "val_mae" or "val_rmse"
    patience=20,               # wait 20 epochs after last improvement
    restore_best_weights=True, # roll back to best model
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,   # 20% of training used for validation
    epochs=1000,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)


y_pred = model.predict(X_test_scaled)


submission = test_df[["id"]].copy()
submission["BeatsPerMinute"] = y_pred

submission.to_csv("submission.csv", index=False)




