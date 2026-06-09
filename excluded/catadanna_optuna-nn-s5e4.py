# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import category_encoders as ce

from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

import optuna

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.callbacks as C
import tensorflow.keras.initializers as I
import tensorflow.keras.activations as A
import tensorflow.keras.metrics as O

from keras.models import load_model


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


df_train["nb_missing_values"] = df_train.isna().sum(axis=1)
df_test["nb_missing_values"] = df_test.isna().sum(axis=1)


LABEL = "Listening_Time_minutes"
FEATURES = [c for c in df_test.columns if c != "id"]
NUM_COLS = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]
CAT_COLS = [c for c in FEATURES if c not in NUM_COLS]

ENCODER = "TE"
IMPUTE = "median"
ALGO = "CB"


if ENCODER == "CAT":
    df_train[CAT_COLS] = df_train[CAT_COLS].fillna("None").astype("category")
    df_test[CAT_COLS] = df_test[CAT_COLS].fillna("None").astype("category")
elif ENCODER == "TE":
    enc = ce.TargetEncoder()
    enc.fit(df_train[CAT_COLS], df_train[LABEL])
    df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS], df_train[LABEL])
    df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS])
elif ENCODER == "OHE":
    df_train_ohe = df_train.copy()
    df_test_ohe = df_test.copy()
    
    df_train_ohe[CATS_TO_ENCODE] = df_train_ohe[CATS_TO_ENCODE].fillna("0000")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df_train_ohe[CATS_TO_ENCODE])
    new_features = ohe.get_feature_names_out()
    
    df_train_ohe[new_features] = 0
    df_train_ohe[new_features] = ohe.transform(df_train_ohe[CATS_TO_ENCODE])
    df_train_ohe = df_train_ohe.drop(columns=CATS_TO_ENCODE)
    
    df_test_ohe[new_features] = 0
    df_test_ohe[new_features] = ohe.transform(df_test_ohe[CATS_TO_ENCODE])
    df_test_ohe = df_test_ohe.drop(columns=CATS_TO_ENCODE)

    FEATURES = [c for c in df_train_ohe.columns if c not in [LABEL, "id"]]    
else:
    enc = ce.OrdinalEncoder()
    enc.fit(df_train[CAT_COLUMNS])
    df_train[CAT_COLUMNS] = enc.transform(df_train[CAT_COLUMNS])
    df_test[CAT_COLUMNS] = enc.transform(df_test[CAT_COLUMNS])    


df_train.loc[df_train["Guest_Popularity_percentage"] > 100, "Guest_Popularity_percentage"] = 100
df_test.loc[df_test["Guest_Popularity_percentage"] > 100, "Guest_Popularity_percentage"] = 100

md = df_train["Episode_Length_minutes"].median()
df_test.loc[df_test["Episode_Length_minutes"] > 150, "Episode_Length_minutes"] = np.nan
df_train.loc[df_train["Episode_Length_minutes"] > 150, "Episode_Length_minutes"] = np.nan

md = df_train["Number_of_Ads"].median()
df_train.loc[df_train["Number_of_Ads"] > 50, "Number_of_Ads"] = np.nan
df_test.loc[df_test["Number_of_Ads"] > 50, "Number_of_Ads"] = np.nan





if ENCODER == "OHE": 
    df_train_ohe = df_train_ohe.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    df_train_ohe = df_train_ohe.fillna(-1)
    df_test_ohe = df_test_ohe.fillna(-1)
else:
    #df_train = df_train.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    df_train = df_train.fillna(-1)
    df_test = df_test.fillna(-1)


X_train, X_val, y_train, y_val = train_test_split(df_train[FEATURES], df_train[LABEL], test_size=0.2, random_state=42)



class OptunaManager():
    def __init__(self):
        self.INPUT_SHAPE = len(FEATURES)
        self.OUTPUT_SHAPE = 1
        self.LABEL = LABEL
        # self.ACTIVATION = L.LeakyReLU(alpha=self.ALPHA_LEAKY_RELU)
        self.ACTIVATION = L.ReLU()
        self.ACTIVATION_OUTPUT = L.ReLU()
        self.LOSS = 'mean_squared_error'
        self.METRICS = O.MeanSquaredError()
        self.OPTIMIZER = "adam"
        self.N_TRIALS = 50
        self.DIRECTION = 'minimize'
        
    def create_model(self, trial):
        # We optimize the numbers of layers, their units and weight decay parameter.
        N_LAYERS = trial.suggest_int("n_layers", 1, 50)
        HIDDEN_LAYER_SIZE = trial.suggest_int("hidden_layers_size", 10, 1000)
        # DROPOUT = trial.suggest_float("dropout", 0.01, 0.5)
        DROPOUT = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
        ACTIVATION = trial.suggest_categorical("activation", ["relu", "linear", "sigmoid", "tanh", "selu", "relu6"])
        OUT_ACTIVATION = trial.suggest_categorical("out_activation", ["relu"])
        
        z = L.Input(shape=(self.INPUT_SHAPE,), name="Id")
        x = L.Dense(HIDDEN_LAYER_SIZE, activation=ACTIVATION, name="d1")(z)
        x = L.BatchNormalization(name="bn1")(x)
        x = L.Dropout(DROPOUT, name="dr1")(x)
        
        if N_LAYERS > 1:
            for i in range(1,N_LAYERS+1):
                dname = str(i+1)
                x = L.Dense(HIDDEN_LAYER_SIZE, activation=ACTIVATION, name="d"+dname)(x)
                x = L.BatchNormalization(name="bn"+dname)(x)
                x = L.Dropout(DROPOUT, name="dr"+dname)(x)
        
        x = L.Dense(self.OUTPUT_SHAPE, activation=OUT_ACTIVATION, name="p1")(x)
        
        model = M.Model(z, x, name="S5E4")
        return model

    def create_optimizer(self, trial):
        # We optimize the choice of optimizers as well as their parameters.
        kwargs = {}
        optimizer_options = [
                             "RMSprop", 
                             "Adam", 
                             "SGD", 
                             #'Nadam',
                             'Lion',
                            # 'Ftrl',
                             'Adamax',
                             'AdamW',
                             #'Adagrad',
                             'Adafactor',
                            # 'Adadelta'
                            ]
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        if optimizer_selected == "RMSprop":
            kwargs["learning_rate"] = trial.suggest_float(
                "RMSprop_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["weight_decay"] = trial.suggest_float("RMSprop_weight_decay", 0.85, 0.99)
            kwargs["momentum"] = trial.suggest_float("RMSprop_momentum", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adam":
            kwargs["learning_rate"] = trial.suggest_float("Adam_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = trial.suggest_float(
                "SGD_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["momentum"] = trial.suggest_float("SGD_momentum", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Nadam":
            kwargs["learning_rate"] = trial.suggest_float("Nadam_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Lion":
            kwargs["learning_rate"] = trial.suggest_float("Lion_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Ftrl":
            kwargs["learning_rate"] = trial.suggest_float("Ftrl_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adamax":
            kwargs["learning_rate"] = trial.suggest_float("Adamax_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "AdamW":
            kwargs["learning_rate"] = trial.suggest_float("AdamW_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adagrad":
            kwargs["learning_rate"] = trial.suggest_float("Adagrad_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adafactor":
            kwargs["learning_rate"] = trial.suggest_float("Adafactor_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adadelta":
            kwargs["learning_rate"] = trial.suggest_float("Adadelta_learning_rate", 1e-5, 1e-1, log=True)

        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
        return optimizer
    
    def objective(self, trial, train_set=(X_train,y_train), valid_set=(X_val, y_val), target=LABEL):
        FILE_PATH = "best_nn.weights.h5"
        
        # Build model and optimizer
        model = self.create_model(trial)
        # select_optimizer = self.create_optimizer(trial)
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=[self.METRICS])
        checkpoint = C.ModelCheckpoint(
                filepath=FILE_PATH,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss', 
                mode='min')

        EPOCHS = trial.suggest_int("epochs", 3, 20)
        BATCHSIZE = trial.suggest_int("batchsize", 128, 4096)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            shuffle=True,
            batch_size=BATCHSIZE,
            epochs=EPOCHS, 
            callbacks=[checkpoint],
            verbose=5,
        )
        
        
        model.load_weights(FILE_PATH)
        pred = model.predict(X_val, batch_size=BATCHSIZE)
        score = mean_squared_error(y_val, pred, squared=False)
        return score


DO_OPTUNA = True


if DO_OPTUNA:
    om = OptunaManager()
    
    study = optuna.create_study(direction=om.DIRECTION)
    study.optimize(om.objective, n_trials=om.N_TRIALS)
    
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    
    print("  Params: ")
    #for key, value in trial.params.items():
    #    print("    {}: {}".format(key, value))
    print(trial.params)    


best_optuna = 13.21473173143074
params_nn =  {'n_layers': 9, 'hidden_layers_size': 492, 'dropout': 0.40552613274040644, 'weight_decay': 6.76537794485094e-05, 'activation': 'selu', 'out_activation': 'relu', 'epochs': 17, 'batchsize': 2080}

best_optuna = 13.151308653403285
params_nn = {'n_layers': 3, 'hidden_layers_size': 397, 'dropout': 0.2, 'weight_decay': 9.602518576943226e-05, 'activation': 'selu', 'out_activation': 'relu', 'epochs': 20, 'batchsize': 180}



if study.best_trial.value < best_optuna:
    print("New best score : ", study.best_trial.value)

