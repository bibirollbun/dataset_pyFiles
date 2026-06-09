# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install tensorflow_text --quiet
!pip install tf-keras --quiet
!pip install --upgrade tensorflow-hub --quiet
!pip install focal_loss --quiet


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'


import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from focal_loss import BinaryFocalLoss


np.random.seed(42)
tf.random.set_seed(42)


df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")


df = df.drop("qid", axis=1)


df.head()


df.isna().sum()


df["target"].plot(kind='hist', title='Target Distribution', figsize=(4,4))


train_df, val_df = train_test_split(df, train_size=0.0075, test_size=0.00075, stratify=df.target.values)


train_df.shape


val_df.shape


train_df["target"].plot(kind='hist', title='Target Distribution', figsize=(4,4))


val_df["target"].plot(kind='hist', title='Target Distribution', figsize=(4,4))


X_train = train_df["question_text"].values
y_train = train_df["target"].values

X_val = val_df["question_text"].values
y_val = val_df["target"].values


plt.figure(figsize=(6, 4))
plt.hist(y_train, bins=2)
plt.xticks([0, 1])
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Class Distribution in Train Data")
plt.show()


plt.figure(figsize=(6, 4))
plt.hist(y_val, bins=2)
plt.xticks([0, 1])
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Class Distribution in Test Data")
plt.show()


preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3")
encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-l-12-h-768-a-12/versions/4", trainable=True)


text_input = keras.layers.Input(shape=(), dtype=tf.string)

# Preprocess the text with TF_HUB
encoder_inputs = preprocessor(text_input)

# Encode the processed text for BERT
outputs = encoder(encoder_inputs)

# Pool the output
pooled_output = outputs["pooled_output"]


x = keras.layers.Dropout(0.5, name="dropout-1")(pooled_output)
x = keras.layers.Dense(16, activation='relu', name="dense-16")(x)

output = keras.layers.Dense(1, activation ='sigmoid', name="output")(x)


model = keras.Model(inputs = [text_input], outputs = [output])


model.summary()


model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])


early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(filepath="/kaggle/working/Best-Quora-Questions-Classifier.h5", monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)


tf.debugging.set_log_device_placement(True)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    verbose=1,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)


model.load_weights("/kaggle/working/Best-Quora-Questions-Classifier.h5")


# Plot training & validation loss
plt.figure(figsize=(12, 5))

# Loss Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Accuracy Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

plt.show()


# Plot training & validation loss
plt.figure(figsize=(12, 5))

# Loss Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['val_loss'], label='Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()

# Accuracy Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.show()


loss, accuracy = model.evaluate(X_val, y_val, batch_size=32)
print(f"Accuracy: {accuracy}, Valudation Loss: {loss}")


best_f1 = best_threshold = 0

y_pred = model.predict(X_val)

for t in np.arange(0.3, 0.8, 0.05):
    y_pred_discrete = [1 if y > t else 0 for y in y_pred]
    f1 = f1_score(y_val, y_pred_discrete)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.2f}")


y_pred = [1 if y > best_threshold else 0 for y in y_pred]

print(classification_report(y_val, y_pred))


X_test = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
X_test = X_test[:10000]


questions = X_test['question_text'].values


predictions = model.predict(questions)

predictions = [1 if y > best_threshold else 0 for y in predictions]


df_submit = pd.DataFrame({
    'qid': X_test['qid'].values,
    'prediction': predictions
})


df_submit.to_csv('/kaggle/working/submission.csv', index=False)

