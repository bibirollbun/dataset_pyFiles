# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()



train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += 2*len(train)
train.y = train.y / train.y.max()
train.y = np.log( train.y )
train.y -= train.y.mean()
train.y *= -1.0

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlim((-5,5))
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        CATS.append(c)
    elif not "age" in c:
        train[c] = train[c].astype("str")
        test[c] = test[c].astype("str")
        CATS.append(c)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


CAT_SIZE = []
CAT_EMB = []
NUMS = []

combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")

for c in FEATURES:
    if c in CATS:
        # LABEL ENCODE
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        n = combined[c].nunique()
        mn = combined[c].min()
        mx = combined[c].max()
        print(f'{c} has ({n}) unique values')

        CAT_SIZE.append(mx+1) 
        CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
            
        m = combined[c].mean()
        s = combined[c].std()
        combined[c] = (combined[c]-m)/s
        combined[c] = combined[c].fillna(0)
        
        NUMS.append(c)
        
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from metric import score

print('TF Version',tf.__version__)


EPOCHS = 4
LRS = [0.01]*2 + [0.001]*1 + [0.0001]*1

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
    x_input_cats = Input(shape=(len(CATS),))
    embs = []
    for j in range(len(CATS)):
        e = tf.keras.layers.Embedding(CAT_SIZE[j],CAT_EMB[j])
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


%%time

REPEATS = 3
FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_nn = np.zeros( len(train) )
pred_nn = np.zeros( len(test) )

#directory = "checkpoints"
#if not os.path.exists(directory):
#    os.makedirs(directory)

for r in range(REPEATS):
    VERBOSE = r==0
    print("#"*25)
    print(f"### REPEAT {r+1} ###")
    print("#"*25)
        
    for i, (train_index, test_index) in enumerate(kf.split(train)):
        
        X_train_cats = train.loc[train_index,CATS].values
        X_train_nums = train.loc[train_index,NUMS].values
        y_train = train.loc[train_index,"y"].values
        y_train2 = train.loc[train_index,"efs"].values
        
        X_valid_cats = train.loc[test_index,CATS].values
        X_valid_nums = train.loc[test_index,NUMS].values
        y_valid = train.loc[test_index,"y"].values
        y_valid2 = train.loc[test_index,"efs"].values
        
        X_test_cats = test[CATS].values
        X_test_nums = test[NUMS].values

        if VERBOSE:
            print(" ","#"*25)
            print(" ",f"### Fold {i+1} ###")
            print(" ","#"*25)
        
        # TRAIN MODEL
        K.clear_session()
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                      loss="mean_squared_error",  
                     )
        v = 2 if VERBOSE else 0
        model.fit([X_train_cats,X_train_nums], [y_train], 
                  validation_data = ([X_valid_cats,X_valid_nums], [y_valid]),
                  callbacks = [lr_callback],
                  batch_size=512, epochs=EPOCHS, verbose=v)
        #model.save_weights(f'{directory}/NN_f{i}_r{r}.weights.h5')
        
        # INFER OOF
        oof_nn[test_index] += model.predict([X_valid_cats,X_valid_nums], verbose=v, batch_size=512).flatten()
        # INFER TEST
        pred_nn += model.predict([X_test_cats,X_test_nums], verbose=v, batch_size=512).flatten()

oof_nn /= REPEATS
pred_nn /= (FOLDS*REPEATS)


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for NN =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_nn
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()




