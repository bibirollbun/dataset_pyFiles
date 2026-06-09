from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Drop ID column
train.drop(columns=["id"], inplace=True)
test_ids = test["id"]
test.drop(columns=["id"], inplace=True)

# Split features and target
X = train.drop("y", axis=1)
y = train["y"]


# Encode categorical features (if any)
X = pd.get_dummies(X, dtype='int')
test = pd.get_dummies(test, dtype='int')
X, test = X.align(test, join="left", axis=1, fill_value=0)


# Assume X, y, test, test_ids already defined.
# Identify which columns are numeric vs binary (0/1)
numeric_cols = [c for c in X.columns if set(X[c].unique()) - {0,1}]
binary_cols = [c for c in X.columns if set(X[c].unique()).issubset({0,1})]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
], remainder='passthrough')  # keep binary columns unchanged

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores, test_preds = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n=== Fold {fold} ===")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    test_prep = preprocessor.transform(test)

    tensorX = tf.convert_to_tensor(X_train_prep, dtype=tf.float32)
    tensorY = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
    tensorXt = tf.convert_to_tensor(X_val_prep, dtype=tf.float32)
    tensorYt = tf.convert_to_tensor(y_val.values, dtype=tf.float32)
    tensorTest = tf.convert_to_tensor(test_prep, dtype=tf.float32)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-6)

    model = models.Sequential([
            layers.Dense(128, activation='swish',
                         kernel_regularizer=regularizers.L1L2(l1=1e-9, l2=1e-5)),
            layers.Dropout(0.3),
        
            layers.Dense(512, activation='relu',
                         kernel_regularizer=regularizers.L1L2(l1=1e-9, l2=1e-5)),
        
            layers.Dense(512, activation='swish',
                         kernel_regularizer=regularizers.L1L2(l1=1e-9, l2=1e-5)),
            layers.Dropout(0.3),
        
            layers.Dense(64, activation='swish',
                         kernel_regularizer=regularizers.L1L2(l1=1e-9, l2=1e-5)),
        
            layers.Dense(1, activation='sigmoid')
        
    ])

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['auc', 'recall', 'accuracy', 'precision'])

    early_stop = EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')

    model.fit(tensorX, tensorY,
              validation_data=(tensorXt, tensorYt),
              epochs=100, batch_size=2048,
              class_weight={0:1,1:12},
              callbacks=[early_stop], verbose=1)

    val_preds = model.predict(tensorXt).ravel()
    test_pred = model.predict(tensorTest).ravel()
    auc_score = roc_auc_score(y_val, val_preds)
    print(f"AUC Fold {fold}: {auc_score:.4f}")

    val_scores.append(auc_score)
    test_preds.append(test_pred)

final_preds = np.mean(test_preds, axis=0)
submission = pd.DataFrame({"id": test_ids, "y": final_preds})
submission.to_csv("submission.csv", index=False)

print("\nCV AUC Scores:", val_scores)
print("Mean AUC:", np.mean(val_scores))



# Final prediction
final_preds = np.mean(test_preds, axis=0)

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "y": final_preds
})
submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv is saved.")
print(f"\nCV AUC Scores: {val_scores}")
print(f"Mean AUC: {np.mean(val_scores):.4f}")


