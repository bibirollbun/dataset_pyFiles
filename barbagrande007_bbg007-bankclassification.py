import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.head()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


ncols = 4
nrows = int(len(train.columns) / ncols)

fig, ax = plt.subplots(nrows, ncols, figsize=(20,20))
ax = ax.flatten()

for i, col in enumerate(train.columns[:-1]):
    sns.histplot(data=train, x=col, ax=ax[i], hue=train.columns[-1], multiple='dodge')

plt.tight_layout()
plt.show()


X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()


def encode_day(X):
    X = X.copy()
    X = pd.DataFrame(X)
    X.columns = ['day']
    return pd.DataFrame({
        'day_sin': np.sin(2 * np.pi * X['day'] / 31),
        'day_cos': np.cos(2 * np.pi * X['day'] / 31)
    })

def encode_month(X):
    X = X.copy()
    X = pd.DataFrame(X)
    X.columns = ['month']
    
    # Map month abbreviations to numeric values
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    X['month_numeric'] = X['month'].map(month_mapping)
    
    return pd.DataFrame({
        'month_sin': np.sin(2 * np.pi * X['month_numeric'] / 12),
        'month_cos': np.cos(2 * np.pi * X['month_numeric'] / 12)
    })


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove day and month since they're handled separately with cyclic encoding
num_cols.remove('day')
cat_cols.remove('month')


transformers = [
    ('cat', OneHotEncoder(), cat_cols),
    ('day_cyclic', FunctionTransformer(encode_day, validate=False), ['day']),
    ('month_cyclic', FunctionTransformer(encode_month, validate=False), ['month']),
    ('scale', StandardScaler(), num_cols)
]

pre_process = ColumnTransformer(transformers, remainder='drop')


X.shape


X = pre_process.fit_transform(X)
X_test = pre_process.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])

early_stopping = EarlyStopping(monitor='val_AUC', mode='max', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_AUC', mode='max', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, 
                    batch_size=256, callbacks=[early_stopping, reduce_lr], verbose=2)



# Plot loss and AUC
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['AUC'], label='Training AUC')
plt.plot(history.history['val_AUC'], label='Validation AUC')
plt.title('AUC over epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()


y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype(int)


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
RocCurveDisplay.from_predictions(y_val, y_pred_proba, ax=ax[0])
ax[0].set_title('ROC Curve')

ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax[1])
ax[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.show()


early_stopping = EarlyStopping(monitor='loss', mode='min', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min', factor=0.5, patience=3, min_lr=1e-6)
model.fit(X, y, epochs=50, batch_size=256, callbacks=[early_stopping, reduce_lr], verbose=2)


results = model.predict(X_test)
submit['y'] = results
submit.to_csv('submission.csv', index=False)

print("Submission file created: 'submission.csv'")
print("AUC on validation set:", roc_auc_score(y_val, y_pred_proba))
print("Model training completed and results saved.")




