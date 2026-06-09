import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

# Keras / TensorFlow
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
gc.collect()


# Set the id to index
train.set_index("id", inplace=True)
test.set_index("id", inplace=True)


train.dropna(subset=["num_sold"], inplace=True)


def process_date_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.day_name()
    #df['week_of_year'] = df['date'].dt.isocalendar().week

    # 周期特征
    df['day_sin']    = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos']    = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    df.drop("date", axis=1, inplace=True)
    return df

train = process_date_features(train)
test  = process_date_features(test)


train.head()


test.head()


X = train.drop(columns=["num_sold"])
y = np.log1p(train["num_sold"])
X_test = test[X.columns]


from sklearn.preprocessing import LabelEncoder

cat_cols = ["country", "store", "product", "day_of_week"]  # 你识别出的类别列

for col in cat_cols:
    le = LabelEncoder()
    # 合并 train & test 一起 fit
    combined = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(combined)

    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])



def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1)  
    ])
    return model



def mape(y_true, y_pred):
    return mean_absolute_percentage_error(np.expm1(y_true), np.expm1(y_pred))

def cross_val_nn_mape(X, y, X_test, n_splits=5, epochs=10, batch_size=1024, verbose=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    test_preds_list = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_mlp(input_dim=X.shape[1])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',  
            metrics=[]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5, verbose=0),
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )

        y_pred_val = model.predict(X_valid).reshape(-1)
        score = mape(y_valid, y_pred_val)
        mape_scores.append(score)

        y_pred_test = model.predict(X_test).reshape(-1)
        test_preds_list.append(y_pred_test)

        print(f"Fold {fold_idx+1} MAPE: {score:.4f}")

    final_test_preds = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), final_test_preds



EPOCHS = 15
BATCH_SIZE = 1024

avg_mape, nn_test_preds = cross_val_nn_mape(
    X=X,
    y=y,
    X_test=X_test,
    n_splits=5,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1  
)

print(f"\nAverage MAPE across folds: {avg_mape:.4f}")

nn_test_preds_expm = np.expm1(nn_test_preds)


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sample_sub["num_sold"] = nn_test_preds_expm
sample_sub.to_csv("submission_nn.csv", index=False)
print("submission_nn.csv saved.")


sample_sub.head()

