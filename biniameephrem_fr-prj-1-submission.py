import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, gc, numpy as np, pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

print(tf.__version__)


def parse_pixels_series(pixels_series: pd.Series) -> np.ndarray:
    """Convert Series of space-separated pixel strings into (N,48,48,1) float32 in [0,1]."""
    arr = np.array([np.fromstring(p, sep=' ') for p in pixels_series], dtype='float32')
    arr = arr.reshape(-1, 48, 48, 1) / 255.0
    return arr

def adapt_to_model_input(x_48_gray: np.ndarray, expected_shape) -> np.ndarray:
    """Resize/grayscale->RGB to match model input shape (H,W,C)."""
    H, W, C = expected_shape  # e.g., (128,128,3)
    x = tf.convert_to_tensor(x_48_gray)      # (N,48,48,1)
    x = tf.image.resize(x, [H, W])           # (N,H,W,1)
    if C == 3:
        x = tf.repeat(x, repeats=3, axis=-1) # (N,H,W,3)
    elif C != 1:
        raise ValueError(f"Unsupported channels in model input: {C}")
    return x.numpy().astype('float32')


# Load the test data into a DataFrame named 'test'.
test_df = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv', dtype=str)
X_test_48 = parse_pixels_series(test_df["pixels"])
print("Raw test tensor:", X_test_48.shape)



# Path to your saved model (.h5 file in Kaggle input folder)
MODEL_PATH = "/kaggle/input/fr_prj_1_model/other/default/1/FR_PRJ_1_model.h5"

# Load the trained model
model = load_model(MODEL_PATH)



print(" Model loaded successfully")
model.summary()



exp_shape = (128,128,3)
X_test = adapt_to_model_input(X_test_48, exp_shape)
print("Preprocessed test:", X_test.shape)



y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Build submission
if "id" in test_df.columns:
    sub = pd.DataFrame({"id": test_df["id"], "emotion": y_pred})
else:
    sub = pd.DataFrame({"id": np.arange(len(y_pred)), "emotion": y_pred})

# Save to the writable folder
save_csv = "/kaggle/working/submission.csv"
sub.to_csv(save_csv, index=False)
print("Saved:", save_csv)




# check working directory files
print(sub.head())
print("Working dir files:", os.listdir("/kaggle/working"))



# Emotion label mapping 
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Pick random 8 images
idxs = np.random.choice(len(X_test_48), 8, replace=False)
plt.figure(figsize=(12, 3))

for i, idx in enumerate(idxs):
    plt.subplot(1, 8, i + 1)
    plt.imshow(X_test_48[idx].squeeze(), cmap='gray')
    plt.title(emotion_labels[int(y_pred[idx])])  # Show emotion name
    plt.axis('off')

plt.tight_layout()
plt.show()

