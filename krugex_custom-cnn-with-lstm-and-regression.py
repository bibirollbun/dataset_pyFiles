# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


print(f"df Train shape {train.shape}")
print(f"df Test shape {test.shape}")


pd.set_option('display.max_columns', None)
train.head()


print("Number of null value in Train DF : ",train.isna().sum().sum())
print("Number of null value in Test DF : ",test.isna().sum().sum())
print("Number of Duplicated Row in Train DF : ", train.duplicated().sum())


X = train.drop(columns='loan_paid_back',axis=1)
y = train['loan_paid_back']


num_cols = train.select_dtypes(exclude='object').drop(columns=['loan_paid_back']).columns
print('Numerical columns :' ,num_cols ,"\n\n number of numerical columns:" ,len(num_cols))

print("="*100)
cat_cols = train.select_dtypes(include= 'object').columns
print('Categorical columns :' ,cat_cols,"\n\n number of categorical columns:" ,len(cat_cols) )


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_train_prep = X_train_prep.toarray()
X_val_prep = X_val_prep.toarray()

# Add the channel dimension for CNN
X_train_cnn = np.expand_dims(X_train_prep, axis=-1)
X_val_cnn = np.expand_dims(X_val_prep, axis=-1)


inputs = keras.Input(shape=(X_train_cnn.shape[1], 1))
x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
x = layers.LayerNormalization()(x)
x = layers.DepthwiseConv1D(kernel_size=3, activation='relu')(x)
x = layers.LayerNormalization()(x)
x = layers.LSTM(64, return_sequences=True)(x)
attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = layers.Add()([x, attn_output])
x = layers.Dense(128, activation='relu')(x)
x = layers.LayerNormalization()(x)
attn_output2 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = layers.Add()([x, attn_output2])
x = layers.Dense(128, activation='relu')(x)
x = layers.LayerNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Build model
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss=keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)


import tensorflow as tf

early_stop = callbacks.EarlyStopping(
    monitor='val_auc', 
    patience=5, 
    restore_best_weights=True,
    mode='max'
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: (1 / neg) * total / 2.0, 1: (1 / pos) * total / 2.0}

history = model.fit(
    X_train_prep, y_train,
    validation_data=(X_val_prep, y_val),
    epochs=5,
    batch_size=512,
    callbacks=[lr_scheduler],
    verbose=1
)


X_test_prep = preprocessor.transform(test)
X_test_prep = X_test_prep.toarray()
X_test_cnn = np.expand_dims(X_test_prep, axis=-1)


train_embeddings = model.predict(X_train_cnn)


test_embeddings  = model.predict(X_test_cnn)


X_train_xgb = np.hstack([X_train_prep, train_embeddings])
X_test_xgb  = np.hstack([X_test_prep,  test_embeddings])


import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="gpu_hist"      # or "gpu_hist"
)

model_xgb.fit(X_train_xgb, y_train, verbose=1)
pred_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]


from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    verbose=False,
    task_type="GPU"
)

cat.fit(X_train_xgb, y_train)
pred_cat = cat.predict_proba(X_test_xgb)[:, 1]


import lightgbm as lgb

lgbm = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=32,
    subsample=0.8,
    colsample_bytree=0.8
)

lgbm.fit(X_train_xgb, y_train)
pred_lgb = lgbm.predict_proba(X_test_xgb)[:, 1]


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_xgb, y_train)

pred_lr = lr.predict_proba(X_test_xgb)[:, 1]


pred_xgb_train = model_xgb.predict_proba(X_train_xgb)[:, 1]
pred_cat_train = cat.predict_proba(X_train_xgb)[:, 1]
pred_lgbm_train = lgbm.predict_proba(X_train_xgb)[:, 1]
pred_lr_train = lr.predict_proba(X_train_xgb)[:, 1]


meta_train = np.column_stack([pred_xgb_train, pred_cat_train, pred_lgbm_train, pred_lr_train])
meta_test  = np.column_stack([pred_xgb, pred_cat, pred_lgb, pred_lr])


from sklearn.linear_model import LogisticRegression

meta = LogisticRegression()
meta.fit(meta_train, y_train)

final_pred = meta.predict_proba(meta_test)[:, 1]


sub['loan_paid_back'] = final_pred
sub.head()


sub.to_csv('submission.csv', index=False)

