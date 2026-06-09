import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer


X_train_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id').copy()
target = 'rainfall'
y_train = X_train_raw.pop(target)


y_train.value_counts().plot.pie(labels=['rain', 'no rain'], autopct='%1.1f%%', shadow=True,
                                       explode=[0.1, 0.1], colors=['lightblue', 'grey'], radius=1.3)
plt.ylabel('');


X_test_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
X_test_raw_ = X_test_raw.ffill()
sub_file = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


X_orig_raw = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv').copy()
y_orig = X_orig_raw.pop(target)

# Remove empty spaces from the features names in original dataset
X_orig_raw.columns = X_orig_raw.columns.str.replace(' ', '')

# Reorder the features in original dataset to match that of competition
X_orig_raw = X_orig_raw[X_train_raw.columns].copy()

# Binarize the target in the original dataset
y_orig = y_orig == 'yes'
# y_orig = y_orig.map({'no': 0, 'yes': 1})

# fill the missing values in test and original datasets
X_orig_raw = X_orig_raw.ffill()

display(X_orig_raw.head(2))


# Decide if features should be engineered
feat_eng = False

def df_processing(df):
    if feat_eng:
        df['pressure'] = df['pressure'] - 1000
        df['temp_gap'] = df['maxtemp'] - df['mintemp']
        df['temp_to_gap_ratio'] = df['temparature']*df['temp_gap']
        for feat in ['temparature', 'dewpoint', 'humidity', 'pressure', 
                     'cloud', 'sunshine', 'windspeed', 'winddirection']:
            df[f'{feat}_previous_day'] = df[feat].shift(1).fillna(0)
            df[f'{feat}_next_day'] = df[feat].shift(-1).fillna(0)
            df[f'{feat}_change_overnight'] = df[feat] - df[f'{feat}_previous_day']
        # # # others
        df['dew_humidity'] = df['dewpoint']*df['humidity']
        df['wind_speeddirection'] = df['windspeed']*df['winddirection']
        df['cloud_windspeed'] = df['cloud']*df['windspeed']
        df['cloud_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_humidity'] = df['cloud']/df['humidity']
        df['temp_to_sunshine'] = df['sunshine']/df['temparature']
        df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
        # # # df['exp_sunshine'] = np.exp(df['sunshine'])
        # # df['log_day'] = np.log(df['day'])
        # df['sin_day'] = np.sin(df['day'])
        # df['wind_deg'] = np.deg2rad(df['winddirection'])
        # df['sin_winddirection'] = np.sin(2*np.pi*df['winddirection'])
        # df['tan_winddirection'] = np.tan(2*np.pi*df['winddirection'])
        # df['day_bins'] = pd.cut(df['day'], bins=12).astype('int')
        df['expected_day'] = df.index%365 + 1
        # df['cloudtest_88'] =  (df.cloud==88).astype(int)
        # df['cloudtest_90'] =  (df.cloud>90).astype(int)
        try:
            df['tan_day'] = np.tan(2*np.pi*df['expected_day']/365)
            df['month'] = pd.cut(df['expected_day'], bins=12, labels=range(1,13)).astype('int')
            df['sin_day']=np.sin(2*np.pi*df['expected_day']/365)
            df['cos_day2']=np.cos(2*np.pi*df['expected_day']/365/2)
            df['cos_winddirection'] = np.cos(2*np.pi*df['winddirection'])

            pass
        except:
            pass
        
      #  df = df.drop(columns=['maxtemp', 'mintemp', 'dewpoint'])
    else:
        df = df
        
    X = df.copy()
    return X
  


X_comb_raw = pd.concat([X_train_raw, X_orig_raw], ignore_index=False)
X_comb_raw


X_train = df_processing(X_train_raw)
X_test = df_processing(X_test_raw_)
X_orig = df_processing(X_orig_raw)
X_comb = df_processing(X_comb_raw)


scaler = MinMaxScaler()

column_trans = make_column_transformer(
    # (OneHotEncoder(), X_train.select_dtypes('object').columns.tolist()),
    (scaler, X_train.select_dtypes('number').columns), 
    remainder='passthrough', 
    sparse_threshold=0)


y_comb = pd.concat([y_train, y_orig], ignore_index=True)


X_train = column_trans.fit_transform(X_train)
X_test = column_trans.transform(X_test)
X_orig = column_trans.transform(X_orig)
X_comb = column_trans.transform(X_comb)


X_test


import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.metrics import AUC
from sklearn.metrics import auc

import warnings
warnings.filterwarnings('ignore')


decision_fnc = 'sigmoid'

model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1], )))
model.add(Dense(200, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))#, kernel_initializer='he_normal'))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Dense(40, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(1, activation=decision_fnc))

model.summary()


# # Compile the model with AUC metric
# model.compile(
#     optimizer=Adam(learning_rate=0.001),  # Or tf.keras.optimizers.Adam(learning_rate=0.001)
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy', AUC(name='roc_auc')]  # Adding AUC metric as 'roc_auc'
# )is


model.compile(optimizer= SGD(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=[AUC(name='auc')])

my_callback = keras.callbacks.EarlyStopping(monitor='auc',min_delta=0.05,patience=10,
                                            verbose=0,mode='max',baseline=None,
                                            restore_best_weights=True,start_from_epoch=180)


include_orig_data = False
if include_orig_data:
    X, y = X_comb, y_comb
else:
    X, y = X_train, y_train





%%time
history = model.fit(X, y, epochs=350, validation_split=1/6, callbacks=[my_callback], shuffle=False)


plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0.5, 1])
plt.legend(loc='upper right')
plt.subplot(122)
plt.plot(history.history['auc'], label='auc')
plt.plot(history.history['val_auc'], label = 'val_auc')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(loc='lower right')

plt.show()


roc_auc_score(y_orig, model.predict(X_orig))


pred_proba = model.predict(X_test)

sub_file[target] = pred_proba

sub_file.to_csv('submission.csv', index=False)
print('The file is ready for submission!')


sub_file.head(10)


plt.subplot(121)
sub_file.rainfall.plot.hist(bins=25, color='lightblue', figsize=(10, 3), 
                          title='Histogram of pred_proba in test set')
plt.xlabel('Predicted Proba')
plt.subplot(122)
(sub_file > 0.5).value_counts().plot.pie(labels=['rain', 'no rain'], autopct='%1.1f%%', shadow=True,
                                       explode=[0.1, 0.1], colors=['lightblue', 'grey'], radius=1.3)
plt.ylabel('');

