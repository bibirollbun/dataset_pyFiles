import pandas as pd, numpy as np
VER=1

# LOAD DATA
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
orig = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
orig = orig.rename({"Gender":"Sex"},axis=1)
orig = orig.rename({"User_ID":"id"},axis=1)
orig['id'] = np.arange( len(orig) ) + 1_000_000

# DISPLAY DATA
print("Train shape, test shape, original shape:")
print(train.shape, test.shape, orig.shape)
train.head()


for df in [train,test, orig]:
    df['Sex'] = df['Sex'].map({'male':0,'female':1}).astype('float32')


FEATURES = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
TARGET = 'Calories'


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

print('TF Version',tf.__version__)


# SIMPLE MLP
def build_model(size=len(FEATURES)):
    x_in = Input(shape=(size,))
    x = Dense(32)(x_in)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Dense(1, activation='linear')(x)
    model = Model(inputs=x_in, outputs=x)
    return model


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

def make_callbacks():
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss',     
        factor=0.5,              
        patience=3,              
        verbose=1,               
        min_lr=1e-6              
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", 
        patience=10,            
        restore_best_weights=True,
        mode="min", 
        verbose=1
    )
    return [lr_callback, early_stop_cb]

EPOCHS = 100


import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
    
    print(f"\n{'#'*28}")
    print(f"{'#'*10} Fold {i+1} {'#'*10}")
    print(f"{'#'*28}")

    # TRAIN
    X_train = train.loc[train_idx,FEATURES].copy()
    y_train = np.log1p( train.loc[train_idx,TARGET] )

    # EXTRA DATA
    for k in range(4):
        X_train = pd.concat([X_train,orig[FEATURES]],axis=0)
        y_train = pd.concat([y_train,np.log1p( orig[TARGET] )],axis=0)

    # VALID
    X_valid = train.loc[valid_idx,FEATURES].copy()
    y_valid = np.log1p( train.loc[valid_idx,TARGET] )

    # TEST
    X_test = test[FEATURES].copy()

    # NORMALIZE FOR NN
    print("Normalizing...", end='')
    norm_cols = [c for c in FEATURES if c not in []]
    means = X_train[norm_cols].mean()
    stds = X_train[norm_cols].std()
    stds = stds.replace(0, 1)
    X_train[norm_cols] = (X_train[norm_cols] - means) / stds
    X_valid[norm_cols] = (X_valid[norm_cols] - means) / stds
    X_test[norm_cols] = (X_test[norm_cols] - means) / stds
    print("done")
    
    start = time.time()

    K.clear_session()
    model = build_model( X_train.shape[1] )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                    loss="mse", 
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                 )
    model.fit(X_train, y_train, 
              validation_data = (X_valid, y_valid),
              callbacks = make_callbacks(),
              batch_size=256, epochs=EPOCHS, verbose=2)

    oof[valid_idx] = model.predict(X_valid,batch_size=512,verbose=2).flatten()
    pred += model.predict(X_test,batch_size=512,verbose=2).flatten()

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

pred /= FOLDS


full_rmse = np.sqrt(mean_squared_error(np.log1p(train[TARGET]), oof))
print(f"Overall CV RMSE: {full_rmse:.5f}")
np.save(f"oof_v{VER}",oof)


mn = train.Calories.min()
mx = train.Calories.max()
test['Calories'] = np.clip( np.expm1( pred ),mn,mx )
test[['id','Calories']].to_csv(f"submission_v{VER}.csv",index=False)
test[['id','Calories']].head()


import matplotlib.pyplot as plt

plt.hist(test['Calories'],bins=100)
plt.title("Test Preds")
plt.show()

