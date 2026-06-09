import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import sigmoid

import keras_tuner as kt


# Load csv
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df = df.drop(['id'], axis=1)
df.head()


# Number of nan values 
df.isna().sum()


# Shape original df vs drop nan
len(df), len(df.dropna())


# Convert string columns in integers
df['Stage_fear'] = (df['Stage_fear'] == 'Yes').astype(int)
df['Drained_after_socializing'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
df['Personality'] = (df['Personality'] == 'Extrovert').astype(int)


# Generate a DataFrame with imputation (using KNNImputer).
imputer = KNNImputer(n_neighbors=1)
imputer_values = imputer.fit_transform(df)

scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(imputer_values)

df = pd.DataFrame(scaler_data, columns=df.columns)


cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 10))

for i, col in enumerate(cols):
    sns.boxplot(data=df, x='Personality', y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_xlabel('Personality')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


print(f"number of rows before removing outliers {len(df)}")
df_filter = pd.DataFrame()

for personality in [0,1]:
    subset = df[df['Personality'] == personality].copy()
    
    for col in cols:
        q1 = subset[col].quantile(0.25)
        q3 = subset[col].quantile(0.75)
        IQR = q3 - q1
        lim_low = q1 - 1.5 * IQR
        lim_upp = q3 + 1.5 * IQR
        
        subset = subset[(subset[col] >= lim_low) & (subset[col] <= lim_upp)]
    
    df_filter = pd.concat([df_filter, subset])

df_filter.reset_index(drop=True)
df = df_filter.copy()
print(f"number of rows after removing outliers {len(df_filter)}")



fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 10))

for i, col in enumerate(cols):
    sns.boxplot(data=df, x='Personality', y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_xlabel('Personality')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


X = df.drop(['Personality'], axis=1)
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))


def model_builder(hp):
    units = hp.Int('units', 64,256, step=16)
    learning_rate = hp.Choice('learning_rate', [0.05, 0.01, 0.005, 0.001])

    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))

    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))


    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


!rm -r /kaggle/working/untitled_project


tuner = kt.RandomSearch(model_builder,
                        objective='val_loss',
                        max_trials=20,
                        executions_per_trial=1)


tuner.search(
    X_train, 
    y_train, 
    epochs=50, 
    validation_split=0.25,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15)])


best_model = tuner.get_best_models(num_models=1)[0]
best_model.evaluate(X_test, y_test)


best_hps = tuner.get_best_hyperparameters(1)[0]
model = model_builder(best_hps)
model.summary()


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.75,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr])


model.evaluate(X_test, y_test)


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()



df_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
X_sub = df_sub.drop(['id'], axis=1)

# Convert string columns in integers
X_sub['Stage_fear'] = (X_sub['Stage_fear'] == 'No').astype(int)
X_sub['Drained_after_socializing'] = (X_sub['Drained_after_socializing'] == 'No').astype(int)

# Generate a DataFrame with imputation (using KNNImputer).
imputer_values = imputer.fit_transform(X_sub)

# Generate predictions
preds = best_model.predict(imputer_values)>.5


output = pd.DataFrame({'id':df_sub['id'], 'Personality': preds.flatten()})
output['Personality'] = output['Personality'].map({True: 'Extrovert', False: 'Introvert'})


output.to_csv('/kaggle/working/output.csv', index=None)
output.head()




