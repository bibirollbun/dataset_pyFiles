import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.utils import class_weight


test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')
test_df.head()


model = tf.keras.models.load_model("/kaggle/input/aptos-blindness-detection-outputs/EfficientNetB5_Regression_UnFrozen.keras")


test_path = "/kaggle/input/aptos2019-blindness-detection/test_images"
def process_test_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [300, 300])
    return image

test_paths = test_df['id_code'].apply(lambda x: os.path.join(test_path, x + ".png")).values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(process_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32)
test_ds = test_ds.prefetch(buffer_size=1)


print("Predicting...")
predictions = model.predict(test_ds, verbose=1)

predictions = predictions.flatten() * 4.0

final_preds = np.clip(np.round(predictions), 0, 4).astype(int)

test_df['diagnosis'] = final_preds
test_df.to_csv('submission.csv', index=False)

print("Done! Submission saved.")


test_df.to_csv("test.csv", index=False)
test_df.to_csv("sample_submission.csv", index=False)
test_df.to_csv("submission.csv", index=False)


test_df.head()

