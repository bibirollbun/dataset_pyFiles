import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from itertools import combinations  
import polars as pl
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import skew, kurtosis

import warnings





def validate_day_alignment(train):
    # Verify that all days now have exactly 6 entries
    fixed_counts = train.groupby('day').size()
    print("Post-Fix Record Counts:", fixed_counts.value_counts())

    # Verify that all sequences are correct
    incorrect_sequences = []
    for day, group in train.groupby('day'):
        expected_ids = [(day - 1) + (365 * i) for i in range(6)]
        actual_ids = sorted(group['id'])
        if expected_ids != actual_ids:
            incorrect_sequences.append(day)

    if incorrect_sequences:
        print(f"ERROR: Some days still have incorrect ID sequences: {incorrect_sequences}")
    else:
        print("All day ID sequences are correctly aligned.")

def fix_day_misalignments(train):
    # Define the reassignment map
    reassignment_map = {
        1132: 38, 1251: 157, 1284: 190, 1290: 196, 1312: 218, 1318: 224, 
        1346: 252, 1352: 258, 1367: 273, 1373: 279, 1380: 286, 1382: 288, 
        1388: 294, 1395: 301, 1400: 306, 1037: 308, 1403: 309, 1404: 310, 
        1406: 312, 1407: 313, 1409: 315, 1414: 320, 1416: 322, 1420: 326, 
        1430: 336, 1438: 344, 1439: 345, 1445: 351, 1452: 358, 1453: 359, 
        1457: 363, 1458: 364, 1459: 365, 1210: 116, 1428: 334
    }

    # Apply the reassignments
    for misplaced_id, correct_day in reassignment_map.items():
        train.loc[train['id'] == misplaced_id, 'day'] = correct_day

    print(train.shape)
    # Verify that all days now have exactly 6 entries
    validate_day_alignment(train)

    return train



train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train = fix_day_misalignments(train)

print("Train shape", train.shape)
train = train.drop_duplicates()

train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


y_test = pd.read_csv('/kaggle/input/lb-probed-rainfall-prediction-validation-set/submission_to_ensemble.csv')

# round predictions to 0 or 1
y_test['rainfall'] = y_test['rainfall'].round().astype(int)

y_test = y_test['rainfall'].values


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


# impute missing values with iterative imputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
train[FEATURES] = imputer.fit_transform(train[FEATURES])
test[FEATURES] = imputer.transform(test[FEATURES])


def engineer_features(df):    

    # **Cyclical Encoding for Day of Year (Important for Periodic Trends)**
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)

    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    
    # Temporal features
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    # Temperature differences
    df['dew_temp_diff'] = df['dewpoint'] - df['temparature']

    # Lagged features
    for feature in ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']:
        for lag in [3, 5, 7]:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    # Rolling mean features
    for feature in ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']:
        for window in [3, 5, 7]:
            df[f'{feature}_roll_mean_{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
    
    # day of year averages 
    for feature in ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']:
        df_group = df.groupby('day')[feature].mean().reset_index()
        col_name = feature + '_day_avg'
        df[col_name] = df['day'].map(dict(zip(df_group['day'], df_group[feature])))

    # wet-bulb temperature
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035

    df['wet_bulb_temp'] = calc_wet_bulb(df['temparature'], df['humidity'])

    # saturated vapor pressure
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))

    df['e_s_temp'] = calc_saturation_vapor_pressure(df['temparature'])
    df['e_s_dewpoint'] = calc_saturation_vapor_pressure(df['dewpoint'])

    # vapor pressure deficit
    df['vapor_pressure_deficit'] = df['e_s_temp'] - df['e_s_dewpoint']

    df['pressure_change_1d'] = df['pressure'] - df['pressure'].shift(1)
    df['humidity_change_1d'] = df['humidity'] - df['humidity'].shift(1)

    df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['cloud_percentage'] = df['cloud'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    df['temp_ratio'] = df['temparature'] / df['maxtemp'].max()

    df['sum_1'] = (df['humidity'] + df['cloud'] + df['dewpoint'])
    df['diff_1'] = (df['cloud'] - df['sunshine']) + df['temparature']

    df['pressure_cut'] = np.where(df['pressure'] < 1000, 1, 0)
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['humidity_dewpoint'] = df['humidity'] * df['dewpoint']
    
    # create interaction features
    features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
    interaction_terms = {}  # Dictionary to store new features

    # Generate pairwise interaction terms
    for feat1, feat2 in combinations(features, 2):
        interaction_terms[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

    # Concatenate all interaction columns at once
    df = pd.concat([df, pd.DataFrame(interaction_terms)], axis=1)

    # backfill nans created by lagging
    df = df.bfill()
    
    # Display the first few rows
    return df

train = engineer_features(train)
test = engineer_features(test)




def fen(df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
    df = df.copy()
    
    rename_dict = {
        'temparature': 'x1', 'humidity': 'x2', 'pressure': 'x3', 'windspeed': 'x4',
        'winddirection': 'x5', 'cloud': 'x6', 'dewpoint': 'x7', 'sunshine': 'x8',
        'pressure_cut': 'x9', 'wet_bulb_temp': 'x10', 'humidity_dewpoint': 'x11', 'vapor_pressure_deficit': 'x12'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Convert to Polars for efficient operations
    df_pl = pl.from_pandas(df)

    df_pl =  df_pl.with_columns(
        _2_1 = ((pl.col('x1')-pl.col('x3'))**2+(pl.col('x2')-pl.col('x4'))**2).sqrt(),
        _2_2 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2).sqrt(),
        _2_3 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2).sqrt(),
        _2_4 = ((pl.col('x1')-pl.col('x9'))**2+(pl.col('x2')-pl.col('x10'))**2).sqrt(),
        _2_5 = ((pl.col('x1')-pl.col('x11'))**2+(pl.col('x2')-pl.col('x12'))**2).sqrt(),
        _2_6 = ((pl.col('x3')-pl.col('x5'))**2+(pl.col('x4')-pl.col('x6'))**2).sqrt(),
        _2_7 = ((pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        _2_8 = ((pl.col('x3')-pl.col('x9'))**2+(pl.col('x4')-pl.col('x10'))**2).sqrt(),
        _2_9 = ((pl.col('x3')-pl.col('x11'))**2+(pl.col('x4')-pl.col('x12'))**2).sqrt(),
        _2_10 = ((pl.col('x5')-pl.col('x7'))**2+(pl.col('x6')-pl.col('x8'))**2).sqrt(),
        _2_11 = ((pl.col('x5')-pl.col('x9'))**2+(pl.col('x6')-pl.col('x10'))**2).sqrt(),
        _2_12 = ((pl.col('x5')-pl.col('x11'))**2+(pl.col('x6')-pl.col('x12'))**2).sqrt(),
        _2_13 = ((pl.col('x7')-pl.col('x9'))**2+(pl.col('x8')-pl.col('x10'))**2).sqrt(),
        _2_14 = ((pl.col('x7')-pl.col('x11'))**2+(pl.col('x8')-pl.col('x12'))**2).sqrt(),
        _2_15 = ((pl.col('x9')-pl.col('x11'))**2+(pl.col('x10')-pl.col('x12'))**2).sqrt(),
        _3_1 = ((pl.col('x1')-pl.col('x4'))**2+(pl.col('x2')-pl.col('x5'))**2+(pl.col('x3')-pl.col('x6'))**2).sqrt(),
        _3_2 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2+(pl.col('x3')-pl.col('x9'))**2).sqrt(),
        _3_3 = ((pl.col('x1')-pl.col('x10'))**2+(pl.col('x2')-pl.col('x11'))**2+(pl.col('x3')-pl.col('x12'))**2).sqrt(),
        _3_4 = ((pl.col('x4')-pl.col('x7'))**2+(pl.col('x5')-pl.col('x8'))**2+(pl.col('x6')-pl.col('x9'))**2).sqrt(),
        _3_5 = ((pl.col('x4')-pl.col('x10'))**2+(pl.col('x5')-pl.col('x11'))**2+(pl.col('x6')-pl.col('x12'))**2).sqrt(),
        _3_6 = ((pl.col('x7')-pl.col('x10'))**2+(pl.col('x8')-pl.col('x11'))**2+(pl.col('x9')-pl.col('x12'))**2).sqrt(),
        _4_1 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2+(pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        _4_2 = ((pl.col('x1')-pl.col('x9'))**2+(pl.col('x2')-pl.col('x10'))**2+(pl.col('x3')-pl.col('x11'))**2+(pl.col('x4')-pl.col('x12'))**2).sqrt(),
        _4_3 = ((pl.col('x5')-pl.col('x9'))**2+(pl.col('x6')-pl.col('x10'))**2+(pl.col('x7')-pl.col('x11'))**2+(pl.col('x8')-pl.col('x12'))**2).sqrt(),
        _5_1 = ((pl.col('x1')-pl.col('x6'))**2+(pl.col('x2')-pl.col('x7'))**2+(pl.col('x3')-pl.col('x8'))**2+(pl.col('x4')-pl.col('x9'))**2+(pl.col('x5')-pl.col('x10'))**2).sqrt(),
    )


    return df_pl.to_pandas()


train_hyper = fen(train, mode="all")
train = pd.concat([train, train_hyper[['_4_1', '_4_2', '_4_3', '_5_1', '_2_1', '_2_2', '_2_3', '_3_1', '_3_2', '_3_3',  '_2_4', '_2_5', '_2_6', '_2_7', '_2_8', '_2_9', '_2_10', '_2_11', '_2_12', '_2_13', '_2_14', '_2_15', '_3_4', '_3_5', '_3_6']]], axis=1)

test_hyper = fen(test, mode="all")
test = pd.concat([test, test_hyper[['_4_1', '_4_2', '_4_3', '_5_1', '_2_1', '_2_2', '_2_3', '_3_1', '_3_2', '_3_3', '_2_4', '_2_5', '_2_6', '_2_7', '_2_8', '_2_9', '_2_10', '_2_11', '_2_12', '_2_13', '_2_14', '_2_15', '_3_4', '_3_5', '_3_6']]], axis=1)




train.shape


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )



# import scaler 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# create a scaler object

scaler = RobustScaler()

# apply the scaler to the numeric columns

train[FEATURES] = scaler.fit_transform(train[FEATURES])

test[FEATURES] = scaler.transform(test[FEATURES])


RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,BatchNormalization, Input, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from keras.regularizers import l2
import keras_tuner as kt
from imblearn.over_sampling import SMOTE
from scipy.stats import rankdata
import warnings
warnings.simplefilter('ignore')



X = train[FEATURES]
y = train['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Creates the autoencoder
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, kernel_regularizer=l2(1e-4))(input_layer)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(128, kernel_regularizer=l2(1e-4))(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(64, kernel_regularizer=l2(1e-4))(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)

    decoded = Dense(128, kernel_regularizer=l2(1e-4))(encoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(256, kernel_regularizer=l2(1e-4))(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Trains autoencoder
autoencoder, encoder = build_autoencoder(X_train.shape[1])
early_stopping_pretraining = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), 
                callbacks=[early_stopping_pretraining], verbose=1)

# Freezes encoder at the beginning of training
encoder.trainable = False  


# Creates the model for the tuner
# Define the HyperModel
def build_model(hp):
    model = Sequential()

    model.add(encoder)
    # Tune number of units in the first and second hidden layers
    model.add(Dense(hp.Int('units_1', min_value=64, max_value=256, step=64), input_dim=X_train.shape[1], kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(hp.Int('units_2', min_value=32, max_value=128, step=32), kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(32, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Dense(16, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Tune learning rate
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log'))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC', 'accuracy'])
    
    return model

# Creates tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuner_dir_1',
    project_name='mlp_tuning'
)

# Early stopping
early_stopping_tuner = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Search for best hypreparameters
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping_tuner])

# Best results found
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hps.values)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []  # Stores models
auc_scores = []  # Stores AUC
histories = []  # Stores the model history

for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)): # X instead of X_train
    print(f"Training fold {fold+1}/5...")
    
    # Train and validation for each fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx] # X instead of X_train
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx] # y instead of y_train
    
    # Creates a new model for each fold
    model = Sequential([
        encoder,  # Pre-trained model (Encoder)
        Dense(best_hps.get('units_1'), kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(best_hps.get('dropout_1')),

        Dense(best_hps.get('units_2'), kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(best_hps.get('dropout_2')),

        Dense(32, kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        Dense(16, kernel_regularizer=l2(1e-4)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Run model
    optimizer = Adam(learning_rate=best_hps.get('learning_rate'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc'), 'accuracy'])

    early_stopping_final = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    
    # Train model in this fold
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=300,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping_final, reduce_lr],
        verbose=1
    )

    # Save model and history
    models.append(model)
    histories.append(history)
    
    # Validate model
    val_auc = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    auc_scores.append(val_auc)
    print(f"Fold {fold+1}: Val AUC = {val_auc:.4f}")

# Average AUC
print(f"Average de AUC folds: {np.mean(auc_scores):.4f}")


test = test[FEATURES]

predictions_all = np.zeros((len(test), len(models)))

for i, model in enumerate(models):
    predictions_all[:, i] = np.squeeze(model.predict(test))

final_predictions = np.mean(predictions_all, axis=1)
final_predictions_class = (final_predictions >= 0.5).astype(int)

test_accuracy = accuracy_score(y_test[:146], final_predictions_class[:146])
test_auc = roc_auc_score(y_test[:146], final_predictions[:146])

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")


colors_val = ['purple', 'blue', 'green', 'gold', 'darkorange']
colors_train = ['violet', 'lightskyblue', 'lightgreen', 'khaki', 'navajowhite']

plt.figure(figsize=(10, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['val_loss'], color=colors_val[i], label=f'Validation Loss Fold {i+1}')
    plt.plot(history.history['loss'], color=colors_train[i], linestyle='dotted', label=f'Training Loss Fold {i+1}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (K-Fold)')
plt.legend()
plt.grid()
plt.show()


from datetime import datetime

# Results DataFrame
result_df = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(final_predictions)),
    "rainfall": final_predictions
})

print(result_df.head())

file_path  = f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
result_df.to_csv(file_path, index=False)




