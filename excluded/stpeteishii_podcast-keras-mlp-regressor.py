import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


train0 = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test0 = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


cols=train0.columns.tolist()
data0=pd.concat([train0,test0],axis=0)


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object':
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


data1=labelencoder(data0)
display(data1.info())


train=data1.iloc[0:len(train0)]
test=data1.iloc[len(train0):]

train=train.dropna()


trainY = train['Listening_Time_minutes']
trainX = train.drop('Listening_Time_minutes',axis=1)
testX = test.drop('Listening_Time_minutes',axis=1)


columns=trainX.columns.to_list()
print(columns)


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
train_oof = np.zeros((trainX.shape[0]))
test_preds = np.zeros((testX.shape[0]))





from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define the MLP architecture using Keras for regression
def build_keras_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(400),
        BatchNormalization(),
        Activation('relu'),
        Dense(100),
        BatchNormalization(),
        Activation('relu'),
        Dense(700),
        BatchNormalization(),
        Activation('relu'),
        Dense(30),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='mse',  # Mean squared error for regression
                 metrics=['mae'])  # Mean absolute error as additional metric
    return model


# Cross-validation loop
for fold_num, (train_index, val_index) in enumerate(kf.split(trainX)):
    print(f"Fitting fold {fold_num+1}/{n_splits}")
    
    # Split data
    train_features = trainX.iloc[train_index]
    train_target = trainY.iloc[train_index]
    val_features = trainX.iloc[val_index]
    val_target = trainY.iloc[val_index]

    # Build model
    model = build_keras_mlp(input_dim=train_features.shape[1])
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train the model
    model.fit(
        train_features, train_target,
        validation_data=(val_features, val_target),
        epochs=1000,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict on validation set
    val_pred = model.predict(val_features).ravel()
    train_oof[val_index] = val_pred
    
    # Evaluation metrics for regression
    mse = mean_squared_error(val_target, val_pred)
    mae = mean_absolute_error(val_target, val_pred)
    r2 = r2_score(val_target, val_pred)
    
    print(f"Fold {fold_num+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Predict on test set (average predictions across folds)
    test_pred_fold = model.predict(testX).ravel()
    test_preds += test_pred_fold / n_splits


test_preds


submit=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
test_preds2=[]
for p in test_preds:
    test_preds2+=[np.clip(p,0,abs(p))]
submit['Listening_Time_minutes']=test_preds2
display(submit)
submit.to_csv('submission.csv',index=False)







