import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(context='notebook')


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


train.info()


CAT_COLS = ['rainfall']
NUM_COLS = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


train.duplicated().sum()


test.duplicated().sum()


train.isna().sum()


test.isna().sum()


test.fillna(test.median(), inplace=True)


rainfall_counts = train['rainfall'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(rainfall_counts, labels=rainfall_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Rainfall')
plt.show()


grouped_data = train.groupby('rainfall')[NUM_COLS].sum()
grouped_data.T.plot(kind='barh', stacked=True, figsize=(12, 8))

plt.xlabel('Rainfall')
plt.ylabel('Numerical Columns')
plt.legend(title='Rainfall', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


for col in NUM_COLS:
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 3, 1)
    train[col].plot.hist(bins=20)
    plt.title(f"Histogram of {col}")
    
    plt.subplot(1, 3, 2)
    stats.probplot(train[col], dist="norm", plot=plt)
    plt.title(f"QQ plot of {col}")
    
    plt.subplot(1, 3, 3)
    sns.boxenplot(x=train[col])
    plt.title(f"Boxen plot of {col}")
    
    plt.tight_layout()
    plt.show()


train.drop(columns=['day'], inplace=True)
test.drop(columns=['day'], inplace=True)


for col in NUM_COLS:
    train[col + '_decile'] = pd.qcut(train[col], 10, labels=False, duplicates='drop')
    test[col + '_decile'] = pd.qcut(test[col], 10, labels=False, duplicates='drop')

train.drop(NUM_COLS, axis=1, inplace=True)
test.drop(NUM_COLS, axis=1, inplace=True)


train.head()


X_train = train.drop(columns=['rainfall'])
y_train = train['rainfall']

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1)
train_history = model.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=64, callbacks=[early_stopping, reduce_lr])


plt.figure(figsize=(20, 10))
plt.plot(train_history.history['accuracy'], label='Train Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


test['rainfall'] = model.predict(test).flatten()
test.to_csv('s5e3-simple-nn-submission.csv', columns=['rainfall'], index=True)

