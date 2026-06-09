""" 
Goal: Apply a simple ANN model on Original Data.
 
Author: Rudra Prasad Bhuyan
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, 
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc
)
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers, regularizers, callbacks
)

from sklearn import set_config
set_config(display="diagram")

import warnings
warnings.simplefilter('ignore')


SUB_PATH    = r"/kaggle/input/playground-series-s5e11/sample_submission.csv" 
TRAIN_PATH  = r"/kaggle/input/playground-series-s5e11/train.csv"
TEST_PATH   = r"/kaggle/input/playground-series-s5e11/test.csv"

RANDOM_SEED = 42
BATCH_SIZE  = 4096
EPOCHS      = 100
VALID_SIZE  = 0.2
MODEL_OUT   = "best_ann.h5"
SUB_OUT     = "submission.csv"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)


TARGET = "loan_paid_back"
ID_COL = "id"

NUMERIC_COLS = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate"
]
CAT_COLS = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]


numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore",))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CAT_COLS)
    ],
    remainder="drop"
)

preprocessor


X_all = train_df[NUMERIC_COLS + CAT_COLS]
y_all = train_df[TARGET].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=VALID_SIZE, 
    random_state=RANDOM_SEED, stratify=y_all
)


preprocessor.fit(X_train)

X_train_p = preprocessor.transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(test_df[NUMERIC_COLS + CAT_COLS])

print("Preprocessed shapes:", X_train_p.shape, X_val_p.shape, X_test_p.shape)


input_dim = X_train_p.shape[1]

def build_model(input_dim,
                l2=1e-5,
                dropout_rate=0.3,
                hidden_units=[256, 128, 64]):
    
    inp = layers.Input(shape=(input_dim,), name="input")
    x = inp
    
    for i, h in enumerate(hidden_units):
        x = layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
            name=f"dense_{i}"
        )(x)
        
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
        
    # Final layers
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2), name="dense_final")(x)
    x = layers.BatchNormalization(name="bn_final")(x)
    x = layers.Dropout(dropout_rate, name="drop_final")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    
    return model


model = build_model(input_dim,
                    l2=1e-4,
                    dropout_rate=0.25,
                    hidden_units=[512, 256, 128])  # modify to make deeper/wider if desired

model.summary()


class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)



es = callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True, verbose=1)
mc = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_auc", mode="max", save_best_only=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4, mode="max", verbose=1)


history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[es, mc, rlr],
    verbose=2
)


val_preds = model.predict(X_val_p, batch_size=8192).ravel()
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC: {val_auc:.5f}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

val_preds = model.predict(X_val_p, batch_size=8192).ravel()

fpr, tpr, thresholds = roc_curve(y_val, val_preds)
roc_auc = auc(fpr, tpr)
print(f"Validation ROC AUC: {roc_auc:.5f}")


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Optional: Save history as CSV
# hist_df = pd.DataFrame(history.history)
# hist_df.to_csv("training_history.csv", index=False)


test_preds = model.predict(X_test_p, batch_size=8192).ravel()

# Create submission file: id + probability
submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    "loan_paid_back_prob": test_preds
})


if sub_df is not None:
    possible_targets = [c for c in sub_df.columns if c != ID_COL]
    if len(possible_targets) == 1:
        submission = pd.DataFrame({
            ID_COL: test_df[ID_COL],
            possible_targets[0]: test_preds
        })

submission.to_csv(SUB_OUT, index=False)
print(f"Saved submission to {SUB_OUT}")


pd.read_csv(r"/kaggle/working/submission.csv").head()

