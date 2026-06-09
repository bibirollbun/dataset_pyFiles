import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.regularizers import l2


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.head()


train.info()


ncols = 5
nrows = int((len(train.columns)) / ncols)

fig, ax = plt.subplots(nrows, ncols, figsize=(20, 10))

ax = ax.flatten()

for idx, col in enumerate(train.columns):
    sns.histplot(data=train, x=col, ax=ax[idx], kde=True)

plt.tight_layout()
plt.show()


rhythmscore_idx = train.nlargest(15000, 'RhythmScore').index
# train = train.drop(rhythmscore_idx)

audioloudness_idx = train.nlargest(50000, 'AudioLoudness').index
# train = train.drop(audioloudness_idx)

vocalcontent_idx = train.nsmallest(150000, 'VocalContent').index
# train = train.drop(vocalcontent_idx)

acousticquality_idx = train.nsmallest(80000, 'AcousticQuality').index
# train = train.drop(acousticquality_idx)

instrumentalscore_idx = train.nsmallest(150000, 'InstrumentalScore').index
# train = train.drop(instrumentalscore_idx)

liveperformance_idx = train.nsmallest(80000, 'LivePerformanceLikelihood').index

lmoodscore_idx = train.nsmallest(20000, 'MoodScore').index

umoodscore_idx = train.nlargest(20000, 'MoodScore').index


combined_idx = list(set().union(rhythmscore_idx, audioloudness_idx, vocalcontent_idx, 
                                acousticquality_idx, instrumentalscore_idx, liveperformance_idx, 
                                lmoodscore_idx, umoodscore_idx))


len(combined_idx)


train = train.drop(combined_idx)


train.shape


iso = IsolationForest(contamination=0.01, random_state=51)
yhat = iso.fit_predict(train)

train_no_outliers = train[yhat == 1]

# train = train[yhat == 1]


train_no_outliers.shape


# Add jitter
# jitter_factor = 0.01

# np.random.seed(52)

# df_jitter = train_no_outliers.copy()

# for col in train_no_outliers.select_dtypes(include=[np.number]).columns:
#     col_std = train_no_outliers[col].std()
#     if col_std > 0:
#         noise = np.random.normal(0, jitter_factor * col_std, size=len(train_no_outliers))
#         df_jitter[col] += noise

# df_augmented = pd.concat([train_no_outliers, df_jitter], ignore_index=True)

df_augmented = train_no_outliers.copy()


X = df_augmented.drop('BeatsPerMinute', axis=1)
y = df_augmented['BeatsPerMinute']


# Plot feature distribution
ncols = 3
nrows = int(np.ceil(len(X.columns) / ncols))

fig, ax = plt.subplots(nrows, ncols, figsize=(14, 7))
ax = ax.ravel()
plt.suptitle('Feature distribution', fontsize=20)

for idx, col in enumerate(X.columns):
    
    sns.histplot(data=X, x=col,ax=ax[idx], kde=True)
    # plt.title(f'{col} distribution')

plt.tight_layout()
plt.show()


# Convert skewed columns
skew_cols = []
normal_cols = []

upper_bound = 1
lower_bound = -1

for col in X.columns:
    skew = X[col].skew().round(2)

    if skew > upper_bound or skew < lower_bound:
        skew_cols.append(col)
    else:
        normal_cols.append(col)


skew_cols


transformer = ColumnTransformer(transformers=[
    ('power', PowerTransformer(method='yeo-johnson'), skew_cols),
    ('scaler', StandardScaler(), normal_cols),
    ],
    remainder='passthrough'
)

preprocess = Pipeline([
    ('transform', transformer),
    # ('poly', PolynomialFeatures(degree=2)),
    # ('pca', PCA(n_components=0.95)),
])


X = preprocess.fit_transform(X)
X_test = preprocess.transform(test)


X.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=51)


model = Sequential([
    
    Input(shape=(X_train.shape[1],)),
    
    Dense(64, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(32, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])


history_dict = history.history

# Plot Loss (MSE)
plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Train Loss (MSE)')
plt.plot(history_dict['val_loss'], label='Val Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training vs Validation Loss')
plt.legend()

# Plot RMSE
plt.subplot(1, 2, 2)
plt.plot(history_dict['root_mean_squared_error'], label='Train RMSE')
plt.plot(history_dict['val_root_mean_squared_error'], label='Val RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Training vs Validation RMSE')
plt.legend()

plt.tight_layout()
plt.show()


model.fit(X, y, epochs=5)


predictions = model.predict(X_test)


sns.kdeplot(predictions, fill=True, label='Predictions')
plt.legend()
plt.show()


submit['BeatsPerMinute'] = predictions


submit.to_csv('BBG007_submission.csv', index=False)




