import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# test data
print(test.shape)
test


# train data
print(train.shape)
train


# Separate numerical and categorical columns

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = train.select_dtypes(include=["object"]).columns

print("Numerical columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))



# Distributions of numerical columns

import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train[col].hist(bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



# Detect & Replace Outliers (Z-score method)

import numpy as np
from scipy import stats

# Copy dataset to avoid modifying original
train_clean = train.copy()

# Exclude 'id' column (not a feature)
num_cols = [c for c in num_cols if c != "id"]

# Set Z-score threshold
threshold = 3

for col in num_cols:
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(train_clean[col]))

    # Print how many outliers detected
    print(f"{col}: {np.sum(z_scores > threshold)} outliers detected")

    # Compute boundaries (mean Â± 3*std)
    mean, std = train_clean[col].mean(), train_clean[col].std()
    upper, lower = mean + threshold * std, mean - threshold * std

    # Replace values outside boundaries with caps
    train_clean[col] = np.where(
        train_clean[col] > upper, upper,
        np.where(train_clean[col] < lower, lower, train_clean[col])
    )

print("\nâœ… Outliers replaced with boundary values (capped).")



# Plot Distributions After Outlier Handling
import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train_clean[col].hist(bins=50)
    plt.title(f"Distribution of {col} (after outlier handling)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



train


# Categorical value counts

for col in cat_cols:
    plt.figure(figsize=(6,4))
    train[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()



# category info
for col in cat_cols:
    print(f"\n---- {col} ----")
    print(train[col].value_counts())



# Mixed Binary + One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Columns that are strictly binary â†’ map directly
binary_cols = ["default", "housing", "loan"]

# for col in binary_cols:
#     train[col] = train[col].map({"no": 0, "yes": 1})
for col in binary_cols:
    train[col] = (
        train[col]
        .astype(str)          # convert to string
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # make lowercase
        .map({"no": 0, "yes": 1})
    )


# Columns that need one-hot encoding
onehot_cols = ["job", "marital", "education", "contact", "month", "poutcome"]

# Apply One-Hot
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(train[onehot_cols])

print("One-hot encoded categorical shape:", X_cat.shape)



print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())

print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())
print(train[binary_cols].head(10))



# Numerical + Target
from sklearn.preprocessing import StandardScaler
import numpy as np

# Exclude id + target
num_features = [c for c in num_cols if c not in ["id", "y"]]

# Scale numerical features
scaler = StandardScaler()
X_num = scaler.fit_transform(train[num_features])

# Combine numerical + categorical
X = np.hstack([X_num, X_cat, train[binary_cols].values])

# Target variable
y = train["y"].values

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)



# Final updated dataframe for training


# Get names for one-hot encoded columns
onehot_feature_names = encoder.get_feature_names_out(onehot_cols)

# Combine all features into a DataFrame
train_updated = pd.DataFrame(
    np.hstack([X_num, X_cat, train[binary_cols].values]),
    columns=list(num_features) + list(onehot_feature_names) + binary_cols
)

# Add target column
train_updated["y"] = y

print("Final dataframe shape:", train_updated.shape)
train_updated.head()



# Model building and training part

# split + class weights

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_val.shape)

# class weights to help with imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
class_weight_dict



# build the ANN
# simple, stable architecture (batchnorm + dropout). tweak widths/dropouts later.

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

tf.random.set_seed(42)


# Building the model
def build_model(input_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

model = build_model(X.shape[1])


# Model summary
model.summary()


# callbacks + training

es = callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=5, restore_best_weights=True
)
rlr = callbacks.ReduceLROnPlateau(
    monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5, verbose=1
)
ckpt = callbacks.ModelCheckpoint(
    "best_ann.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=2048,
    class_weight=class_weight_dict,
    callbacks=[es, rlr, ckpt],
    verbose=1
)


# save the model
model.save("ann_v1.h5")  # or .keras


# evaluation

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# best model is already restored by EarlyStopping
val_pred_proba = model.predict(X_val, verbose=0).ravel()
val_pred = (val_pred_proba >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_val, val_pred_proba))
print(classification_report(y_val, val_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))



# plot training curves

import matplotlib.pyplot as plt

def plot_hist(h, key):
    plt.figure(figsize=(6,4))
    plt.plot(h.history[key], label=f"train_{key}")
    plt.plot(h.history[f"val_{key}"], label=f"val_{key}")
    plt.title(key)
    plt.xlabel("epoch"); plt.ylabel(key)
    plt.legend(); plt.show()

for k in ["loss", "accuracy", "auc", "precision", "recall"]:
    if k in history.history:
        plot_hist(history, k)



# test predictions (no labels)

# numeric
X_test_num = scaler.transform(test[num_features])

# binary yes/no â†’ 0/1
test_bin = test[binary_cols].replace({"no": 0, "yes": 1}).values

# one-hot categorical
X_test_cat = encoder.transform(test[onehot_cols])

# final test design matrix
X_test = np.hstack([X_test_num, X_test_cat, test_bin])

# predict probabilities
test_proba = model.predict(X_test, verbose=0).ravel()

# if competition wants 0/1 instead of probabilities:
test_pred = (test_proba >= 0.5).astype(int)

# build dataframe
submission = pd.DataFrame({
    "id": test["id"],
    "y": test_proba   # or test_pred
})
submission.to_csv("submission.csv", index=False)



# ANN v2: deeper, GELU, L2 regularization, tuned dropout + batch size

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

tf.random.set_seed(42)


# Build improved ANN
def build_model(input_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(
        256, activation="gelu",
        kernel_regularizer=regularizers.l2(1e-5)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(
        128, activation="gelu",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        64, activation="gelu",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(
        32, activation="gelu",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# Build model
model = build_model(X.shape[1])
model.summary()


# Callbacks
es = callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=5, restore_best_weights=True
)
rlr = callbacks.ReduceLROnPlateau(
    monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5, verbose=1
)
ckpt = callbacks.ModelCheckpoint(
    "best_ann_v2.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1
)


# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=1024,  # smaller than before for better generalization
    class_weight=class_weight_dict,
    callbacks=[es, rlr, ckpt],
    verbose=1
)



# save the model
model.save("ann_v2.h5")  # or .keras


# Model Evaluation

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Validation predictions
val_pred_proba = model.predict(X_val, verbose=0).ravel()
val_pred = (val_pred_proba >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_val, val_pred_proba))
print(classification_report(y_val, val_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))

# ğŸ”¹ Plot Confusion Matrix
cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Training Curves

def plot_hist(h, key, ymax=None):
    plt.figure(figsize=(6,4))
    plt.plot(h.history[key], label=f"train_{key}")
    plt.plot(h.history[f"val_{key}"], label=f"val_{key}")
    plt.title(f"Training Curve: {key}")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    if ymax: plt.ylim(0, ymax)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

for k in ["loss", "accuracy", "auc", "precision", "recall"]:
    if k in history.history:
        ymax = 1.05 if k != "loss" else None
        plot_hist(history, k, ymax)



# Test Predictions

# Numeric features
X_test_num = scaler.transform(test[num_features])

# Binary yes/no â†’ 0/1
test_bin = test[binary_cols].replace({"no": 0, "yes": 1}).values

# One-hot categorical (same encoder as training)
X_test_cat = encoder.transform(test[onehot_cols])

# Final design matrix
X_test = np.hstack([X_test_num, X_test_cat, test_bin])

# Predict
test_proba = model.predict(X_test, verbose=0).ravel()
test_pred = (test_proba >= 0.5).astype(int)

# Build submission
submission = pd.DataFrame({
    "id": test["id"],
    "y": test_proba   # use probabilities (most competitions want this)
})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission saved as submission.csv")


