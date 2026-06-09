import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Target and feature columns
target_col = 'target'
feature_cols = [
    'seconds_in_bucket', 'imbalance_buy_sell_flag', 'imbalance_size',
    'matched_size', 'bid_size', 'ask_size', 'reference_price', 'far_price',
    'near_price', 'ask_price', 'bid_price', 'wap'
]


df = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')



print(df.shape)
df.head()



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(df[target_col], bins=50, color='skyblue')
plt.title('Target Variable Distribution')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


df[feature_cols].describe().T



for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique}")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
corr_matrix = df[feature_cols + [target_col]].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Feature Correlation with Target")
plt.show()



# nullの合計を求める
null_counts = df.isnull().sum()
print("Null values count per column:")
print(null_counts)

# データの総量を求める (行数と列数)
total_rows = df.shape[0]
total_columns = df.shape[1]
print(f"\nTotal number of rows: {total_rows}")
print(f"Total number of columns: {total_columns}")



df_null_far_price = df[df['far_price'].isnull()]
df_null_far_price


df


df['imbalance_buy_sell_flag'].value_counts(normalize=True).plot(kind='bar')
plt.title("Imbalance Buy/Sell Flag Distribution")
plt.xticks(rotation=0)
plt.show()


import seaborn as sns

key_features = ['wap', 'bid_size', 'ask_size', 'imbalance_size', 'reference_price','seconds_in_bucket']
for col in key_features:
    sns.scatterplot(x=df[col], y=df[target_col])
    plt.title(f'{col} vs Target')
    plt.show()



df['bucket'] = pd.cut(df['seconds_in_bucket'], bins=[0, 200, 400, 600])
df.groupby('bucket')[target_col].mean().plot(kind='bar')
plt.title('Average Target Value by Time Bucket')
plt.ylabel('Mean Target')
plt.show()



# Optionally bin seconds into 10-second intervals for cleaner boxplot
df['seconds_bin'] = (df['seconds_in_bucket'] // 10) * 10

plt.figure(figsize=(12, 6))
sns.boxplot(x='seconds_bin', y='target', data=df, palette='Blues')
plt.title('Target Distribution Across Auction Timeline')
plt.xlabel('Seconds in Bucket (Binned)')
plt.ylabel('Target')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['target'], bins=100, kde=True, color='darkblue')
plt.title('Distribution of Target (Price Movement)')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()



# Now this works
df.dropna(subset=[target_col])

df.fillna(df.mean(), inplace=True)

df = df.dropna(subset=feature_cols + [target_col])
X = df[feature_cols]
y = df[target_col]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


cat_model = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, verbose=0)
cat_model.fit(X_train, y_train)
cat_val_preds = cat_model.predict(X_val)


def build_gru_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)
    output = Dense(1)(x)
    return Model(inputs, output)

# Reshape for sequence model: (samples, time_steps=1, features)
X_train_seq = np.expand_dims(X_train.values, axis=1)
X_val_seq = np.expand_dims(X_val.values, axis=1)

gru_lstm_model = build_gru_lstm_model(X_train_seq.shape[1:])
gru_lstm_model.compile(optimizer='adam', loss='mae')
gru_lstm_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
                   epochs=5, batch_size=512, callbacks=[EarlyStopping(patience=2)])
gru_lstm_val_preds = gru_lstm_model.predict(X_val_seq).flatten()


from sklearn.metrics import mean_absolute_error

stacked_val_preds = 0.5 * cat_val_preds + 0.5 * gru_lstm_val_preds
print(f"Stacked MAE: {mean_absolute_error(y_val, stacked_val_preds):.5f}")



counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    test.fillna(test.mean(), inplace=True)
    if counter == 0:
        print(test.head(3))
        print(revealed_targets.head(3))
        print(sample_prediction.head(3))
    sample_prediction['target'] = 0
    env.predict(sample_prediction)
    counter += 1




