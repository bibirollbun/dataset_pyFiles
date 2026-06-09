import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

print('TF Version',tf.__version__)


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')  
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')  

test = pd.read_csv('//kaggle/input/playground-series-s5e2/test.csv') 

train_combined = pd.concat([train,train_extra],axis=0,ignore_index=True)


train_combined.head(5)


RMV = ['id', 'Price']
FEATURES = [c for c in train_combined.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


# Identify categorical and numerical columns
CATEGORICAL = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
NUMERICAL = ['Compartments', 'Weight Capacity (kg)']


# Features (X) and target (y)
#train_combined = df.drop(columns=['id', 'Price'])  # 'Price' is the target column
# = df['Price']


for c in CATEGORICAL:
    if train_combined[c].dtype=="object":
        train_combined[c] = train_combined[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")

    elif train_combined [c].dtype!="object":
        train_combined[c] = train_combined[c].astype("str")
        test[c] = test[c].astype("str")

print(f"In these features, there are {len(CATEGORICAL)} CATEGORICAL FEATURES: {CATEGORICAL}")





CATEGORICAL_SIZE = []
CATEGORICAL_EMB = []
NUMS = []

combined_all = pd.concat([train_combined,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")

for c in FEATURES:
    if c in CATEGORICAL:
        # LABEL ENCODE
        combined_all[c],_ = combined_all[c].factorize()
        combined_all[c] -= combined_all[c].min()
        combined_all[c] = combined_all[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        n = combined_all[c].nunique()
        mn = combined_all[c].min()
        mx = combined_all[c].max()
        print(f'{c} has ({n}) unique values')

        CATEGORICAL_SIZE.append(mx+1) 
        CATEGORICAL_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
    else:
        if combined_all[c].dtype=="float64":
            combined_all[c] = combined_all[c].astype("float32")
        if combined_all[c].dtype=="int64":
            combined_all[c] = combined_all[c].astype("int32")
            
        m = combined_all[c].mean()
        s = combined_all[c].std()
        combined_all[c] = (combined_all[c]-m)/s
        combined_all[c] = combined_all[c].fillna(0) # try other things here
        
        NUMS.append(c)
        
train = combined_all.iloc[:len(train_combined)].copy()
test = combined_all.iloc[len(train_combined):].reset_index(drop=True).copy()


# Display the first few rows
train.head()


EPOCHS = 15
LRS = [0.01]*10 + [0.001]*3 + [0.0001]*2

def lrfn(epoch):
    return LRS[epoch]

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
        format(lr_y[0], max(lr_y), lr_y[-1]))
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)


def build_model():
    
    # CATEGORICAL FEATURES
    x_input_cats = Input(shape=(len(CATEGORICAL),))
    embs = []
    for j in range(len(CATEGORICAL)):
        e = tf.keras.layers.Embedding(CATEGORICAL_SIZE[j],CATEGORICAL_EMB[j])
        x = e(x_input_cats[:,j])
        x = tf.keras.layers.Flatten()(x)
        embs.append(x)
        
    # NUMERICAL FEATURES
    x_input_nums = Input(shape=(len(NUMS),))
    
    # COMBINE
    x = tf.keras.layers.Concatenate(axis=-1)(embs+[x_input_nums]) 
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[x_input_cats,x_input_nums], outputs=x)
    
    return model


#early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
#callbacks = [early_stopping, reduce_lr]


%%time
REPEATS = 3
FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_nn = np.zeros( len(train) )
pred_nn = np.zeros( len(test) )

oof_losses = []
oof_mses = []


for r in range(REPEATS):
    VERBOSE = r == 0
    print("#" * 25)
    print(f"### REPEAT {r + 1} ###")
    print("#" * 25)

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        # Prepare training and validation data
        X_train_cats = train.loc[train_index, CATEGORICAL].values
        X_train_nums = train.loc[train_index, NUMS].values
        y_train = train.loc[train_index, "Price"].values

        X_valid_cats = train.loc[test_index, CATEGORICAL].values
        X_valid_nums = train.loc[test_index, NUMS].values
        y_valid = train.loc[test_index, "Price"].values

        X_test_cats = test[CATEGORICAL].values
        X_test_nums = test[NUMS].values

        if VERBOSE:
            print(" ", "#" * 25)
            print(" ", f"### Fold {i + 1} ###")
            print(" ", "#" * 25)

        # Train the model
        K.clear_session()
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss="mean_squared_error",
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])  # Add RMSE as a metric
        v = 2 if VERBOSE else 0
        history = model.fit([X_train_cats, X_train_nums], [y_train],
                            validation_data=([X_valid_cats, X_valid_nums], [y_valid]),
                            callbacks=[lr_callback],
                            batch_size=512, epochs=EPOCHS, verbose=v)

        # Store the loss and MSE from the training history
        oof_losses.append(history.history["val_loss"][-1])  # Last epoch's validation loss
        oof_mses.append(history.history["val_root_mean_squared_error"][-1])  # Last epoch's validation MSE

        # INFER OOF
        oof_nn[test_index] += model.predict([X_valid_cats, X_valid_nums], verbose=v, batch_size=512).flatten()
        # INFER TEST
        pred_nn += model.predict([X_test_cats, X_test_nums], verbose=v, batch_size=512).flatten()

# Average OOF predictions across repeats
oof_nn /= REPEATS
pred_nn /= (FOLDS * REPEATS)




# Compute overall OOF MSE
oof_mse = mean_squared_error(train["Price"], oof_nn)

# Print evaluation results
print("\nOverall OOF MSE:", oof_mse)
print("Average Validation Loss across folds and repeats:", np.mean(oof_losses))
print("Average Validation MSE across folds and repeats:", np.mean(oof_mses))


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.prediction = pred_nn
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

