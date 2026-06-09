import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from othello_board_visualizer import OthelloBoardVisualizer


train_df = pd.read_csv("/kaggle/input/evaluate-othello-boards/train.csv")
external_df = pd.read_csv("/kaggle/input/othello-board-evaluations/board_evaluations.csv")
df = pd.concat([train_df, external_df], ignore_index=True)

MATRIX_COLUMNS = [f'cell_{i}' for i in range(64)]
X = df[MATRIX_COLUMNS].to_numpy(dtype='float32')
X_turn_player = df["turn_player"].to_numpy(dtype='float32')
y = df["turn_player_advantage"].to_numpy(dtype='float32')


X = X * X_turn_player[:, np.newaxis]
X = X.reshape(-1, 8, 8, 1)
y = y / 64.0


TURN_PLAYER_MAP = {1: "Black", -1: "White"}
EXAMPLE_TURN_PLAYER = TURN_PLAYER_MAP[X_turn_player[0]]
EXAMPLE_PLOT_TITLE = f"Advantage for {EXAMPLE_TURN_PLAYER}: {y[0] * 64.0}"

OthelloBoardVisualizer().plot_board_state(X[0], EXAMPLE_PLOT_TITLE)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])


OPTIMIZER = Adam(learning_rate=0.0002)
model.compile(optimizer=OPTIMIZER, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50)


plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()


val_loss, val_rmse = model.evaluate(X_val, y_val)
print(f"Validation RMSE (original scale): {val_rmse * 64:.2f}")


test_df = pd.read_csv("/kaggle/input/evaluate-othello-boards/test.csv")

X_test = test_df[MATRIX_COLUMNS].to_numpy(dtype='float32')
X_test_turn_player = test_df["turn_player"].to_numpy(dtype='float32')

X_test = X_test * X_test_turn_player[:, np.newaxis]
X_test = X_test.reshape(-1, 8, 8, 1)

y_pred_scaled = model.predict(X_test).flatten()
y_pred_original = y_pred_scaled * 64.0


submission = pd.DataFrame({
    "id": test_df["id"],
    "turn_player_advantage": y_pred_original
})

submission.to_csv("submission.csv", index=False)
submission.head()

