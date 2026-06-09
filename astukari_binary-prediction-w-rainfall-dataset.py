
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_probability as tfp

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
serving_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

train_df.head(10)


input_features = list(train_df.columns)
input_features.remove("id")
input_features.remove("day")
input_features.remove("rainfall")

print(f"Input features: {input_features}")


# Convert dataframes to tensors
features_train = tf.convert_to_tensor(train_df[input_features].values, dtype=tf.float32)

labels_train = tf.convert_to_tensor(train_df['rainfall'].values, dtype=tf.float32)

features_serving = tf.convert_to_tensor(serving_df[input_features].values, dtype=tf.float32)


# # Specify model
# model = tfp.glm.Bernoulli()

# # Fit model on training data
# coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
#     model_matrix=features_train,
#     response=labels_train, 
#     model=model)

# # Make predictions on serving data
# predictions = model.predict(tf.matmul(features_serving, coeffs))

# # Print first 10 predictions
# print("\nFirst 10 predictions (probabilities):")
# print(predictions[:10])

# # Calculate accuracy on training data
# train_predictions = model.predict(tf.matmul(features_train, coeffs))
# train_predictions_binary = tf.cast(train_predictions > 0.5, tf.float32)
# accuracy = tf.reduce_mean(tf.cast(train_predictions_binary == labels_train, tf.float32))

# # Calculate log loss on training data
# epsilon = 1e-15  # Small constant to avoid log(0)
# log_loss = -tf.reduce_mean(
#     labels_train * tf.math.log(train_predictions + epsilon) + 
#     (1 - labels_train) * tf.math.log(1 - train_predictions + epsilon)
# )

# print(f"\nTraining Accuracy: {accuracy:.4f}")
# print(f"Training Log Loss: {log_loss:.4f}")


# Prepare the features and target
X_train = train_df[input_features].values
y_train = train_df['rainfall'].values

X_serving = serving_df[input_features].values

# Normalize the features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_norm = (X_train - mean) / std
X_serving_norm = (X_serving - mean) / std

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(len(input_features),))
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_norm,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Make predictions on serving data
predictions = model.predict(X_serving_norm)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': serving_df['id'],
    'rainfall': predictions.flatten()
})

#fix the infamous 2707 value
faulty_id = [2707]
submission_df.loc[submission_df.id.isin(faulty_id), "rainfall"] = '0.5'


# Save predictions
submission_df.to_csv('submission.csv', index=False)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



