import numpy as np
import pandas as pd
import os
import tensorflow_hub as hub
import tensorflow_text as text
import tf_keras as keras
import tensorflow as tf

from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
for item in os.walk("/kaggle/input/quora-insincere-questions-classification"):
    print(item)

path_list = list(list(os.walk("/kaggle/input/quora-insincere-questions-classification"))[0])


del path_list[1]

print(path_list)


sample_submission = pd.read_csv(os.path.join(path_list[0], path_list[1][0]))
len(sample_submission), sample_submission.head()


train_df = pd.read_csv(os.path.join(path_list[0], path_list[1][2])).drop("qid", axis=1)
len(train_df), train_df.head()


train_df.isna().sum()


train_df["target"].plot(kind='hist', title='Target Distribution', figsize=(4,4))


np.random.seed(13)
tf.random.set_seed(13)


from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train_df, train_size=0.008, test_size=0.002, stratify=train_df.target.values)


len(train_data), train_data.head(), train_data.shape


len(val_data), val_data.head(), val_data.shape


X_train, y_train = train_data['question_text'].values, train_data['target'].values
X_val, y_val = val_data['question_text'].values, val_data['target'].values


X_train[:5], y_train[:5]


X_val[:5], y_val[:5]


# Create the preprocessing layer
preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")

# Create the encoder layer
encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/en-uncased-l-12-h-768-a-12/4", trainable=True)

# Build the model using the functional API
text_input = keras.layers.Input(shape=(), dtype=tf.string)
encoder_inputs = preprocessor(text_input)
outputs = encoder(encoder_inputs)
pool_output = outputs["pooled_output"]

x = keras.layers.Dropout(0.5)(pool_output)
x = keras.layers.Dense(16, activation='relu')(x)

output = keras.layers.Dense(1, activation='sigmoid')(x)


model = keras.Model(inputs=[text_input], outputs=[output])


print(model.summary())


model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5), loss=keras.losses.BinaryCrossentropy(
    label_smoothing=0.0,
    axis=-1,
    reduction="sum_over_batch_size",
    name="binary_crossentropy"), metrics=["accuracy"])


early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(filepath="/kaggle/working/Best-Quora-Questions-Classifier.weights.h5", monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)


history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs=5,
    verbose=1,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)


model.load_weights("/kaggle/working/Best-Quora-Questions-Classifier.weights.h5")


import matplotlib.pyplot as plt

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


model.evaluate(X_val, y_val, batch_size=32)


from sklearn.metrics import f1_score

best_f1 = best_threshold = 0

y_pred = model.predict(X_val)

for t in np.arange(0.3, 0.8, 0.05):
    y_pred_discrete = [1 if y > t else 0 for y in y_pred]
    f1 = f1_score(y_val, y_pred_discrete)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.2f}")


from sklearn.metrics import classification_report

y_pred = [1 if y > best_threshold else 0 for y in y_pred]

print(classification_report(y_val, y_pred))


X_test = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
len(X_test)


questions = X_test["question_text"].values
len(questions)


predictions = model.predict(questions[:1000])

predictions = [1 if y > best_threshold else 0 for y in predictions]


df_submit = pd.DataFrame({
    'qid': X_test["qid"][:1000].values,
    'prediction': predictions
})

df_submit.to_csv('/kaggle/working/submission.csv', index=False)




