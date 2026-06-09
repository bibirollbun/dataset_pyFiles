import pandas as pd
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


from sklearn.preprocessing import LabelEncoder

def preprocessing(df, categorical_columns):

    df_processed = df.copy()
    df_processed['is_episode_length_min'] = df_processed['Episode_Length_minutes'].notnull().astype(int)
    df_processed['is_guest_popularity'] = df_processed['Guest_Popularity_percentage'].notnull().astype(int)
 
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
    
    return df_processed


categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day','Publication_Time', 'Episode_Sentiment']
    
train_processed = preprocessing(train, categorical_cols)
test_processed = preprocessing(test, categorical_cols)



train_processed = train_processed.drop(columns=categorical_cols)
test_processed = test_processed.drop(columns=categorical_cols )


# Impute NaN with mean
mean_episode_length = train_processed['Episode_Length_minutes'].mean()
train_processed['Episode_Length_minutes'].fillna(mean_episode_length, inplace=True)
test_processed['Episode_Length_minutes'].fillna(mean_episode_length, inplace=True)

# Impute Guest_Popularity_percentage with mean
mean_guest_popularity = train_processed['Guest_Popularity_percentage'].mean()
train_processed['Guest_Popularity_percentage'].fillna(mean_guest_popularity, inplace=True)
test_processed['Guest_Popularity_percentage'].fillna(mean_guest_popularity, inplace=True)



train_processed = train_processed.drop(columns='id')
test_processed = test_processed.drop(columns='id')


test_id = test_processed['id']


from sklearn.model_selection import train_test_split

X = train_processed.drop(columns=['Listening_Time_minutes'])
y = train_processed['Listening_Time_minutes']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state ensures reproducibility
)



most_common_ads = X_train['Number_of_Ads'].mode()[0]
X_train['Number_of_Ads'].fillna(most_common_ads, inplace=True)


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# 1. Scale features (recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_processed)

# 2. Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression (no activation)
])

# 3. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 4. Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()



X_test_scaled


y_pred = model.predict(X_test_scaled)


pred_df = pd.DataFrame(y_pred, columns=['Listening_Time_minutes'])


result_df = pd.concat([test_id.reset_index(drop=True), pred_df], axis=1)

print(result_df.head())


result_df.to_csv('submission.csv', index=False)

