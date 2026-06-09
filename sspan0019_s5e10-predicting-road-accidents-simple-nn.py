import pandas as pd
import numpy as np

train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')


train_data.info()


test_data.info()


CAT_COLS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']


NUM_COLS = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']


train_data.duplicated().sum()


train_data.drop_duplicates(inplace=True)


train_data.isna().sum()


test_data.isna().sum()


import matplotlib.pyplot as plt
import seaborn as sns

for col in CAT_COLS:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    train_data[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], title=col)
    train_data.groupby(col)['accident_risk'].mean().plot.bar(ax=ax[1], title='Average Accident Risk')
    plt.show()


import scipy.stats as stats

for col in NUM_COLS:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    train_data[col].plot.hist(bins=20)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 3, 2)
    stats.probplot(train_data[col].dropna(), dist="norm", plot=plt)
    plt.title(f"QQ plot of {col}")

    plt.subplot(1, 3, 3)
    sns.boxplot(x=train_data[col])
    plt.title(f'Boxen plot of {col}')

    plt.tight_layout()
    plt.show()


corr = train_data[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask)


BINARY_CAT_COL = ['road_signs_present', 'public_road', 'holiday', 'school_season']

for col in BINARY_CAT_COL:
    train_data[col] = train_data[col].map({True: 1.0, False: 0.0})
    test_data[col] = test_data[col].map({True: 1.0, False: 0.0})


from sklearn.preprocessing import OneHotEncoder

NOMINAL_CAT_COLS = ['road_type', 'lighting', 'weather']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_cols = encoder.fit_transform(train_data[NOMINAL_CAT_COLS])
encoded_col_names = encoder.get_feature_names_out(NOMINAL_CAT_COLS)
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=train_data.index)
train_data = pd.concat([train_data.drop(columns=NOMINAL_CAT_COLS), encoded_df], axis=1)

encoded_cols_test = encoder.transform(test_data[NOMINAL_CAT_COLS])
encoded_df_test = pd.DataFrame(encoded_cols_test, columns=encoded_col_names, index=test_data.index)
test_data = pd.concat([test_data.drop(columns=NOMINAL_CAT_COLS), encoded_df_test], axis=1)


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['morning', 'afternoon', 'evening']], handle_unknown='use_encoded_value', unknown_value=-1)
train_data['time_of_day'] = encoder.fit_transform(train_data[['time_of_day']])
test_data['time_of_day'] = encoder.transform(test_data[['time_of_day']])


from sklearn.preprocessing import MinMaxScaler

uniform_features = ['num_lanes', 'speed_limit']

min_max_scaler = MinMaxScaler()
train_data[uniform_features] = min_max_scaler.fit_transform(train_data[uniform_features])
test_data[uniform_features]  = min_max_scaler.transform(test_data[uniform_features])


from sklearn.preprocessing import FunctionTransformer

skewed_features = ['num_reported_accidents']

train_data[skewed_features] = np.log1p(train_data[skewed_features])
test_data[skewed_features] = np.log1p(test_data[skewed_features])


train_data.info()


train_data.isna().sum()


train_data.head()


import tensorflow as tf
from tensorflow import keras

target = 'accident_risk'
predictors = train_data.drop(columns=[target])
X_train, X_test = predictors.align(test_data, join="outer", axis=1, fill_value=0)
y_train = train_data[target]

model = keras.Sequential([
    keras.layers.InputLayer(shape=(predictors.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error',
    metrics=['root_mean_squared_error'])


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=3, 
    min_lr=1e-9)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True)

train_history = model.fit(
    X_train, 
    y_train, 
    validation_split=0.3, 
    epochs=100, 
    batch_size=32, 
    callbacks=[reduce_lr, early_stopping])


X_test.drop(columns=['accident_risk'], inplace=True, errors='ignore')
predictions = model.predict(X_test)
X_test['accident_risk'] = predictions
X_test[['accident_risk']].to_csv('submission.csv', index=True)

