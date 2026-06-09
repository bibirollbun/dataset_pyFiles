import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")
print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
train.head(3)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


TARGET = 'ev'
X = train.drop([TARGET, "id"], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop(columns='id').copy()


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_pred_nn = np.zeros(len(X))
test_preds_nn = np.zeros((len(X_test), FOLDS))
fold_mse_nn = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"Training fold {fold} ...")
    
    X_train = X.iloc[train_idx]
    X_val   = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val   = y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    nn_val_preds_list = []
    nn_test_preds_list = []
    
    for rep in range(3):
        seed = 42 + rep
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        nn_model = Sequential()
        nn_model.add(Dense(256, input_dim=X_train_scaled.shape[1]))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(256))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(256))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(256))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(128))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(128))
        nn_model.add(BatchNormalization())
        nn_model.add(Activation('relu'))
        
        nn_model.add(Dense(1, activation='linear'))

        nn_model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0)
        nn_model.fit(X_train_scaled, y_train, 
                     validation_data=(X_val_scaled, y_val),
                     epochs=1000,
                     batch_size=32,
                     callbacks=[early_stop],
                     verbose=0)
        
        nn_val_pred_rep = nn_model.predict(X_val_scaled).flatten()
        nn_test_pred_rep = nn_model.predict(X_test_scaled).flatten()
        nn_val_preds_list.append(nn_val_pred_rep)
        nn_test_preds_list.append(nn_test_pred_rep)
    
    nn_val_pred = np.mean(nn_val_preds_list, axis=0)
    nn_test_pred = np.mean(nn_test_preds_list, axis=0)
    
    mse_nn = mean_squared_error(y_val, nn_val_pred)
    fold_mse_nn.append(mse_nn)
    oof_pred_nn[val_idx] = nn_val_pred
    
    print(f"Fold {fold} MSE (NN): {mse_nn:.8f}")
    
    test_preds_nn[:, fold - 1] = nn_test_pred

overall_nn_mse = mean_squared_error(y, oof_pred_nn)
print("\nOverall OOF MSE (NN):")
print(f"  NN = {overall_nn_mse:.8f}")

final_test_pred_nn = test_preds_nn.mean(axis=1)
print("\nFinal NN test predictions (first 10 samples):")
print(final_test_pred_nn[:10])


sub = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv")
sub.ev = final_test_pred_nn
sub.to_csv("submission.csv", index=False)
sub.head(3)

