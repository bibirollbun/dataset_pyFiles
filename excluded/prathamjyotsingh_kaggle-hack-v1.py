import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks


# Load datasets
train_path = "/kaggle/input/thapar-kaggle-hack-v-01/X_train.csv"
test_path = "/kaggle/input/thapar-kaggle-hack-v-01/X_test.csv"
submission_path = "/kaggle/input/thapar-kaggle-hack-v-01/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(submission_path)

# Store 'id' column for test data if available
if "id" in df_test.columns:
    test_ids = df_test["id"]
    df_test = df_test.drop(columns=["id"])
else:
    test_ids = None

# Drop 'id' column from train if present
if "id" in df_train.columns:
    df_train = df_train.drop(columns=["id"])

# Separate features and target variable
X = df_train.drop(columns=["target"])
y = df_train["target"]
X_test = df_test.copy()

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = encoder.fit_transform(X[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

# Standardize numerical features
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# Combine processed numerical and categorical features
X_processed = np.hstack((X_num, X_cat))
X_test_processed = np.hstack((X_test_num, X_test_cat))

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Convert labels to categorical (One-hot encoding for Neural Network)
num_classes = len(np.unique(y))
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

# Define Neural Network (MLP) Model
model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01) ,input_shape=(X_train.shape[1],)),
    keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(num_classes, activation='softmax')  # Output layer (Multiclass classification)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=5,          # Stop after 5 epochs of no improvement
    restore_best_weights=True  # Restore best model weights
)
# Train the model
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=100,
                    batch_size=32,
                    verbose=1, 
                    callbacks=[early_stopping])

# Evaluate model
val_loss, val_accuracy = model.evaluate(X_val, y_val_cat)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Train on full data
y_full_cat = keras.utils.to_categorical(y, num_classes)
model.fit(X_processed, y_full_cat, epochs=50, batch_size=32, verbose=1)

# Make predictions on test data
y_test_pred = np.argmax(model.predict(X_test_processed), axis=1)

# Create submission file
df_submission["target"] = y_test_pred

# Ensure 'id' column is not duplicated before inserting
if test_ids is not None and "id" not in df_submission.columns:
    df_submission.insert(0, "id", test_ids)

df_submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")




