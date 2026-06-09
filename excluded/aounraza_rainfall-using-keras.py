import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import callbacks
from sklearn.metrics import roc_auc_score, auc, roc_curve , accuracy_score , confusion_matrix
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


df


df.columns


df.info()


test.info()


test['winddirection']=test['winddirection'].fillna(test['winddirection'].median())


test.info()


df.describe().T


df.isnull().sum()


df.nunique()


df


cols=df.columns
cols=cols.drop(['rainfall'])
for i in cols:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y=i, x='rainfall')
    plt.subplot(1,2,2)
    sns.kdeplot(data=df, x=i, hue='rainfall')
    plt.show()


df.columns


pair=df[['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity','cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']]
plt.figure(figsize=(10,10))
sns.pairplot(pair, hue='rainfall', corner=True)
plt.show()


corr=pair.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap='Spectral', annot=True)
plt.show()


df


test


x_train=df.drop(columns=['id','day', 'rainfall'])
x_test=test.drop(columns=['id','day'])
y_train=df['rainfall']


s=StandardScaler()
x_train[x_train.columns]=s.fit_transform(x_train[x_train.columns])


x_test[x_test.columns]=s.fit_transform(x_test[x_test.columns])


x_train


# model=keras.Sequential([
#     layers.BatchNormalization(input_shape=[10]),
#     layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
#     layers.Dropout(rate=0.3),
#     layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
#     layers.BatchNormalization(),
#     layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
#     layers.Dropout(0.2),
#     layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])


# model=keras.Sequential([
#     layers.BatchNormalization(input_shape=[10]),
#     layers.Dense(256, kernel_initializer='he_normal'),
#     layers.LeakyReLU(alpha=0.01),
#     layers.Dropout(0.4),

#     layers.Dense(128, kernel_initializer='he_normal'),
#     layers.BatchNormalization(),
#     layers.LeakyReLU(alpha=0.01),

#     layers.Dense(64, kernel_initializer='he_normal'),
#     layers.BatchNormalization(),
#     layers.LeakyReLU(alpha=0.01),
#     layers.Dropout(0.3),

#     layers.Dense(32, kernel_initializer='he_normal'),
#     layers.LeakyReLU(alpha=0.01),
#     layers.Dense(1, activation='sigmoid')
# ])


def build_model(hp):
    model=keras.Sequential()

    model.add(layers.Input(shape=(x_train.shape[1],)))
    model.add(layers.Dense(units=hp.Int(name='hidden_layer',
                                       min_value=200,
                                       max_value=512),
                          activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=keras.regularizers.L2(hp.Float(name='regularizer',
                                                                           min_value=0.001,
                                                                           max_value=0.015,
                                                                           default=0.005))))

    for i in range(hp.Int(name='additional_layers', min_value=1, max_value=3)):
        model.add(layers.Dense(units=hp.Int(name='hidden_layer_'+str(i+2),
                                           min_value=32,
                                           max_value=200),
                              activation='relu',
                              kernel_initializer='he_normal'))
        model.add(layers.Dropout(rate=hp.Float(name='dropout_'+str(i+2),
                                              min_value=0.35,
                                              max_value=0.6,
                                              step=0.07)))
        model.add(layers.BatchNormalization())


    model.add(layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC()])

    return model


tuner=kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective('val_auc', 
                          direction='max'),
    max_trials=30,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=3,
    max_consecutive_failed_trials=2,
    overwrite=True
)


early_stopping=callbacks.EarlyStopping(monitor='val_auc',
                                      patience=14,
                                      verbose=0,
                                      restore_best_weights=True)
reduce_lr=callbacks.ReduceLROnPlateau(monitor='val_auc',
                                     patience=4,
                                     verbose=1,
                                     min_lr=0.000001,
                                     factor=0.4)


tuner.search(x_train,
            y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[reduce_lr, early_stopping])


best_trials=tuner.oracle.get_best_trials(num_trials=5)
for i in best_trials:
    print(i.hyperparameters.values)
    print()


best_model=tuner.get_best_models(num_models=5)


preds=best_model[4].predict(x_test).flatten()


sub['rainfall']=preds


sub.to_csv('submission.csv', index=False)


# early_stopping=callbacks.EarlyStopping(
#     min_delta=0.001,
#     patience=30,
#     restore_best_weights=True
# )


# model.compile(
#     optimizer='rmsprop',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# history=model.fit(
#     x_train,
#     y_train,
#     batch_size=150,
#     epochs=50,
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=1
# )


# history_df=pd.DataFrame(history.history)


# history_df.loc[5:,['loss', 'val_loss']].plot()


# history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")


# predictions=model.predict(x_test).flatten()


# predictions


# sub['rainfall']=predictions


# sub.to_csv('submission.csv', index=False)



# print(sub.head())
# print(sub.columns)
# print(sub.shape)


# sub




