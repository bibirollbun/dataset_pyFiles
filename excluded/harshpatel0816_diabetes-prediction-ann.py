import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df.head()


df.drop(columns=["id"],inplace=True)


df.head()


df.info()


df.isna().sum()


df.duplicated().sum()


df.describe()


df['diagnosed_diabetes'].value_counts()


df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
df["bmi_age"] = df["bmi"] * df["age"]
df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1)


X = df.drop(columns=["diagnosed_diabetes"])
y = df['diagnosed_diabetes']


cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)



n_features = X_train_processed.shape[1]
print(n_features)


X_train.shape, y_train.shape, X_test.shape, y_test.shape


model = Sequential([
    Dense(256, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.25),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])


model.summary()


loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
    gamma=2.0,
    alpha=0.4
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=[tf.keras.metrics.AUC(name="auc")]
)



class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

class_weights = {0: class_weights[0], 1: class_weights[1]}


early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=50,
    batch_size=1024,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])


plt.plot(history.history["val_auc"])
plt.plot(history.history["auc"])


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


test_ids = test_df["id"]


test_df["bp_ratio"] = test_df["systolic_bp"] / (test_df["diastolic_bp"] + 1)
test_df["chol_ratio"] = test_df["ldl_cholesterol"] / (test_df["hdl_cholesterol"] + 1)
test_df["bmi_age"] = test_df["bmi"] * test_df["age"]
test_df["activity_bmi"] = test_df["physical_activity_minutes_per_week"] / (test_df["bmi"] + 1)


X_test = test_df.drop(columns=["id"])


X_test_processed = preprocessor.transform(X_test)


test_preds = model.predict(X_test_processed).ravel()


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})


submission.head()


submission.to_csv("submission.csv",index=False)

