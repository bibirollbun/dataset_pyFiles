import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


train = pd.concat([train, train_extra])



print(train.shape, test.shape)


print(train.dtypes)


CAT_COLS = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
NUM_COLS = ['Compartments', 'Weight Capacity (kg)']


print('TRAIN')

for col in CAT_COLS:
    print(f'{col}: {train[col].nunique()} unique values')

print()

print('TEST')

for col in CAT_COLS:
    print(f'{col}: {test[col].nunique()} unique values')



print('TRAIN')

for col in CAT_COLS:
    unique_values = train[col].dropna().unique()
    print(f'{col}: {sorted(unique_values)}')

print()

print('TEST')

for col in CAT_COLS:
    unique_values = test[col].dropna().unique()
    print(f'{col}: {sorted(unique_values)}')


print('TRAIN')

for col in CAT_COLS:
    print(f'{col}: {train[col].isnull().mean():.4f}%')

print()

print('TEST')

for col in CAT_COLS:
    print(f'{col}: {test[col].isnull().mean():.4f}%')


print(train[NUM_COLS].describe())


print('TRAIN')

for col in NUM_COLS:
    print(f'{col}: {train[col].nunique()} unique values')

print()

print('TEST')

for col in NUM_COLS:
    print(f'{col}: {test[col].nunique()} unique values')


print('TRAIN')
print(f"Compartments: {sorted(train['Compartments'].dropna().unique())}")

print()

print('TEST')
print(f"Compartments: {sorted(test['Compartments'].dropna().unique())}")


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


print('TRAIN')

for col in NUM_COLS:
    print(f'{col}: {train[col].isnull().mean():.4f}%')

print()

print('TEST')

for col in NUM_COLS:
    print(f'{col}: {test[col].isnull().mean():.4f}%')


import matplotlib.pyplot as plt

for col in NUM_COLS:
    train[col].plot(kind='hist', title=col)
    plt.show()


for col in NUM_COLS:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f'{col}: {train[(train[col] < lower_bound) | (train[col] > upper_bound)].shape[0]} outliers')


train[CAT_COLS] = train[CAT_COLS].fillna('unknown')
test[CAT_COLS] = test[CAT_COLS].fillna('unknown')


from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=5)
knn.fit(train[NUM_COLS].sample(frac=0.05))

train[NUM_COLS] = knn.transform(train[NUM_COLS])
test[NUM_COLS] = knn.transform(test[NUM_COLS])


train['Weight Capacity (kg)'] = pd.cut(train['Weight Capacity (kg)'], bins=[0, 10, 20, 30, 100], labels=[1, 2, 3, 4])
test['Weight Capacity (kg)'] = pd.cut(test['Weight Capacity (kg)'], bins=[0, 10, 20, 30, 100], labels=[1, 2, 3, 4])

train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].astype(int)
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].astype(int)


train = pd.get_dummies(train, columns=CAT_COLS)
test = pd.get_dummies(test, columns=CAT_COLS)


train.head()


train.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

X_train = train.drop(columns=['Price'])
y_train = train['Price']

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

train_history = model.fit(X_train, y_train, 
                    validation_split=0.2,
                    epochs=50,
                    batch_size=1024, 
                    callbacks=[early_stopping, reduce_lr])


plt.figure(figsize=(10, 5))
plt.plot(train_history.history['loss'], label='train loss')
plt.plot(train_history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


X_test = test
y_pred = model.predict(X_test)

test['Price'] = y_pred
test[['Price']].to_csv('s5e2-submission.csv', index=True)

