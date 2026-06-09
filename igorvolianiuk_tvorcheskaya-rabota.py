import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s4e4/train.csv",index_col="id")
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']


train_df.shape


train_df.info()


print(train_df.isna().sum())


train_df.describe(include="all")


train_df["Sex"].unique()


target = "Rings"

num_cols = [
    "Length", "Diameter", "Height",
    "Whole weight", "Whole weight.1", "Whole weight.2",
    "Shell weight"
]


correlation_matrix = train_df[num_cols + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


vc = train_df.Rings.value_counts()
plt.figure(figsize=(10, 4))
plt.bar(vc.index, vc)
plt.title("Train data")
plt.show()


for c in num_cols:
    plt.figure()
    plt.hist(train_df[c], bins=50)
    plt.title(f"Distribution: {c}")
    plt.xlabel(c)
    plt.ylabel("Count")
    plt.show()


for c in num_cols:
    plt.figure()
    plt.boxplot(train_df[c].dropna(), vert=True)
    plt.title(f"Boxplot: {c}")
    plt.ylabel(c)
    plt.show()


def clip_outliers_iqr(df, cols, k=1.5):
    df = df.copy()
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        df[c] = df[c].clip(lo, hi)
    return df

train_df = clip_outliers_iqr(train_df, num_cols, k=1.5)


for c in num_cols:
    plt.figure()
    plt.boxplot(train_df[c].dropna(), vert=True)
    plt.title(f"Boxplot: {c}")
    plt.ylabel(c)
    plt.show()


def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def print_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = rmsle(y_true, y_pred)
    print(prefix)
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("RMSLE:", r)


X = train_df[num_cols].copy()
y = train_df[target].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

model_num = keras.Sequential([
    normalizer,
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model_num.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse"
)


X_train


model_num.build((None, len(num_cols)))
model_num.summary()


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )
]


history = model_num.fit(
    X_train, y_train_log,
    validation_data=(X_val, y_val_log),
    epochs=200,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)


def plot_loss(history):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title('Обучение модели MSE')
    plt.legend()
    plt.show()

plot_loss(history)


val_pred_log = model_num.predict(X_val).ravel()
val_pred = np.expm1(val_pred_log)

print_metrics(y_val.values, val_pred, prefix="Только числовые признаки")


X2 = pd.get_dummies(train_df[["Sex"] + num_cols], columns=["Sex"], drop_first=False)
X2 = X2.astype(np.float32)

X2_train, X2_val, y2_train, y2_val = train_test_split(
    X2, train_df["Rings"], test_size=0.2, random_state=42
)

y2_train_log = np.log1p(y2_train).astype(np.float32)
y2_val_log   = np.log1p(y2_val).astype(np.float32)

normalizer2 = layers.Normalization(axis=-1)
normalizer2.adapt(X2_train.values)

model_all = keras.Sequential([
    normalizer2,
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model_all.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")


X2_train


model_all.build((None, X2_train.shape[1]))
model_all.summary()


history2 = model_all.fit(
    X2_train.values, y2_train_log.values,
    validation_data=(X2_val.values, y2_val_log.values),
    epochs=200,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)


plot_loss(history2)


val_pred2_log = model_all.predict(X2_val).ravel()
val_pred2 = np.expm1(val_pred2_log)

print_metrics(y2_val.values, val_pred2, prefix="Все признаки")


from sklearn.model_selection import KFold

def build_model(input_df_for_adapt):
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(input_df_for_adapt))

    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

cv = KFold(n_splits=5, shuffle=True, random_state=0)

scores = []
oof_pred = np.zeros(len(y))

for fold_i, (train_index, valid_index) in enumerate(cv.split(X, y), start=1):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    model = build_model(X_train)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train_log,
        validation_data=(X_valid, y_valid_log),
        epochs=200,
        batch_size=256,
        callbacks=callbacks,
        verbose=0
    )

    val_pred_log = model.predict(X_valid, verbose=0).ravel()
    val_pred = np.expm1(val_pred_log)
    val_pred = np.maximum(val_pred, 0)

    oof_pred[valid_index] = val_pred

    print_metrics(y_valid.values, val_pred, prefix=f"Fold {fold_i}")
    scores.append(val_pred)

print_metrics(y.values, oof_pred, prefix="CV (OOF) Только числовые признаки")

