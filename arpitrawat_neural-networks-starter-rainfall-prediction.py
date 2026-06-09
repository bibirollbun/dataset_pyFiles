import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.shape,test.shape


train.info()


test.info()


test['winddirection'].fillna(test['winddirection'].mean(), inplace=True)


from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, BatchNormalization
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale only feature columns, keeping 'rainfall' unchanged
train[FEATURES] = scaler.fit_transform(train[FEATURES])

# Transform the test set and convert it back to a DataFrame
test = pd.DataFrame(scaler.transform(test[FEATURES]), columns=FEATURES)



FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)

oof = np.zeros(len(train))  # Out-of-fold predictions
pred = np.zeros(len(test))  # Test set predictions

val_losses = []  # Store validation losses for each fold
val_aucs = []  # Store validation AUC scores for each fold

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 30)
    print(f"### Fold {i+1}")
    print("#" * 30)

    # Splitting data
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "rainfall"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "rainfall"]
    x_test = test[FEATURES].copy()

    # Defining the Neural Network model
    model = Sequential([
        Dense(64, input_dim=len(FEATURES), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        PReLU(),
        Dense(1, activation='sigmoid')  # Sigmoid for probability output
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(name='auc')])

    # Early Stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )

    # Training the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=100, verbose=0,
        callbacks=[early_stop]  # Apply early stopping
    )

    # Store last epoch's validation loss & AUC
    fold_loss = history.history['val_loss'][-1]
    fold_auc = history.history['val_auc'][-1]
    val_losses.append(fold_loss)
    val_aucs.append(fold_auc)

    print(f"Fold {i+1} - Validation Loss: {fold_loss:.4f}, Validation AUC: {fold_auc:.4f}")

    # INFER OOF (Out-of-Fold Predictions)
    oof[test_index] = model.predict(x_valid).flatten()

    # INFER TEST (Accumulate predictions)
    pred += model.predict(x_test).flatten()

# COMPUTE AVERAGE TEST PREDICTIONS
pred /= FOLDS  # Averaging over folds

# PRINT FINAL AVERAGES
print("\n#### FINAL RESULTS ####")
print(f"Avg Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
print(f"Avg Validation AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")



model.summary()


submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.rainfall=pred
submission.to_csv("submission.csv", index=False)




