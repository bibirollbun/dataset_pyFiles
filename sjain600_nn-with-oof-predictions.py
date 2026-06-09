# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.simplefilter('ignore')
# Ignore only the specific FutureWarning from pandas option
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    message=r".*use_inf_as_na option is deprecated.*"
)

# Setting matplotlib defaults
plt.rc('figure', figsize=(8, 5), dpi=120)

plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=15, titlepad=10)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv', index_col='id')


train.sample(5)


train.info()


num_cols = train.select_dtypes(include=['int64']).columns
train[num_cols] = train[num_cols].astype('int8')
test[num_cols] = test[num_cols].astype('int8')


train.info()


train.shape


test.shape


train.isnull().sum()


test.isnull().sum()


train.describe()


test.describe()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train.copy()
y = X.pop('ev')
X_test = test.copy()

n_folds=10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=34)

oof_dnn = np.zeros(len(y))
test_preds_dnn = np.zeros((len(X_test), n_folds))
fold_mse_dnn = []



from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


oof_pred_dnn = np.zeros(len(X))
test_preds_dnn = np.zeros((len(X_test), n_folds))
fold_mse_dnn = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) 

    def build_model():
        return Sequential([
            Input(shape=(X_train_scaled.shape[1],)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='leaky_relu'),
            Dense(1)
        ])

    val_preds, test_preds = [], []
    for rep in range(3):
        np.random.seed(34 + rep)
        tf.random.set_seed(34 + rep)
        
        model = build_model()
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=50, restore_best_weights=True)],
            verbose=0
        )
        
        val_preds.append(model.predict(X_val_scaled).flatten())
        test_preds.append(model.predict(X_test_scaled).flatten()) 

    fold_val_pred = np.mean(val_preds, axis=0)
    fold_test_pred = np.mean(test_preds, axis=0)
    
    fold_mse = mean_squared_error(y_val, fold_val_pred)
    fold_mse_dnn.append(fold_mse)
    oof_pred_dnn[val_idx] = fold_val_pred
    test_preds_dnn[:, fold-1] = fold_test_pred
    
    print(f"Fold {fold} MSE: {fold_mse:.9f}")

final_test_pred_dnn = test_preds_dnn.mean(axis=1)
print(f"\nOverall OOF MSE: {mean_squared_error(y, oof_pred_dnn):.8f}")
print("Test predictions (first 10):", final_test_pred_dnn[:10])


sub = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv')
sub['ev'] = final_test_pred_dnn 
sub.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

