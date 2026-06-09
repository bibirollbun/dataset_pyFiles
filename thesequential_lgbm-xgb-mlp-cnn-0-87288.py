import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import math
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from sklearn.utils import class_weight
import xgboost
print(xgboost.__version__)

# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv") 
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')  


#Check for correlation
plt.figure(figsize=(12, 10))
sns.heatmap(data=train.corr(), annot=True, linewidths=0.2);


#check values
train.head()


np.cos(train['winddirection'])


#check for possible zeros
train.describe()


original['day']


#Check NaN in training
original_missing_pct = train.isnull().mean()
print("train missing percentages (fraction):")
print(original_missing_pct*100)


#Check NaN in test
original_missing_pct = test.isnull().mean()
print("test missing percentages (fraction):")
print(original_missing_pct*100)
    


#Check NaN in test
original_missing_pct = original.isnull().mean()
print("Original missing percentages (fraction):")
print(original_missing_pct*100)
    


original.info()


original.rename(columns={'humidity ': 'humidity'}, inplace=True)
original.rename(columns={'pressure ': 'pressure'}, inplace=True)
original.rename(columns={'cloud ': 'cloud'}, inplace=True)
original.rename(columns={'         winddirection': 'winddirection'}, inplace=True)
original['rainfall'] = original['rainfall'].map({'yes': 1, 'no': 0})
original['rainfall'].unique()
original['rainfall'] = original['rainfall'].astype(int)
original['day'] = np.arange(1, len(original) + 1)


#check for unique values
from pprint import pprint
pprint({col:len(train[col].unique()) for i,col in enumerate(train.columns)}) 


#create new features
from sklearn.preprocessing import PolynomialFeatures

def feat_eng(df):

    #add month information
    df['month'] = df['day']/30

    #add temp range information
    df['temp_range'] = df['maxtemp']-df['mintemp']

    # Convert wind direction from degrees to radians
    df['wind_direction_rad'] = np.deg2rad(df['winddirection'])

    # Compute the u and v components
    df['wind_u'] = df['windspeed'] * np.cos( np.deg2rad(df['winddirection']))
    df['wind_v'] = df['windspeed'] * np.sin( np.deg2rad(df['winddirection']))
    
    #dewpoint depression
    df['dewpoint_depression'] = df['temparature'] - (df['dewpoint'])

    #sun coverage ratio
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-3)

    #difference in pressure from consecutive days
    df['pressure_diff'] = df['pressure'].diff()
    df['temp_diff'] = df['temparature'].diff()

    #uncorrelated variables
    df['wind_humidity'] = df['windspeed'] * (df['humidity'])
    df['dewpoint_humidity'] = df['humidity'] * (df['dewpoint'])
    df['humidity_maxtemp'] = df['maxtemp'] * (df['humidity'])

    
   

    lag_cols = ['pressure', 'temparature', 'humidity', 'cloud', 'sunshine']
    for col in lag_cols:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # ========== rolling windows ==========

    #rolling_window = 3
    #for col in lag_cols:
        #df[f'{col}_rollmean{rolling_window}'] = df[col].rolling(rolling_window).mean()



     #drop day info and id 
    if 'id' in df.columns: 
      df = df.drop(columns=['day','id']).copy()
    else:
      df = df.drop(columns=['day']).copy()
    
    df = df.drop(columns=['windspeed','winddirection']).copy()
    #df = df.drop(columns=['cloud','humidity']).copy()

    return df
    


# Impute numerical columns: fill NaN with mean from training data
for c in train.columns:
    if c != 'rainfall':
        train[c]= train[c].fillna(train[c].median())
        test[c]= test[c].fillna(test[c].median())
# Impute numerical columns of original dataset: fill NaN with mean from training data
for c in original.columns:
    if c != 'rainfall':
       original[c]= original[c].fillna(original[c].median())

dataset_train = feat_eng(train)
dataset_test = feat_eng(test)
dataset_original = feat_eng(original)


# Impute numerical columns: fill NaN with mean from training data
for c in dataset_train.columns:
    if c != 'rainfall':
       dataset_train[c]= dataset_train[c].fillna(dataset_train[c].median())
       dataset_test[c]=dataset_test[c].fillna(dataset_test[c].median())

# Impute numerical columns of original dataset: fill NaN with mean from training data
for c in dataset_original.columns:
    if c != 'rainfall':
        dataset_original[c]= dataset_original[c].fillna(dataset_original[c].median())



# Reordering columns
dataset_original = dataset_original[dataset_train.columns]

# Define target and features columns
target = ['rainfall']
features = [col for col in dataset_train.columns if col not in target]


# Scale features
sc = MaxAbsScaler()
dataset_train[features] = sc.fit_transform(dataset_train[features])
dataset_test[features] = sc.transform(dataset_test[features])
dataset_original[features] = sc.transform(dataset_original[features])



#combine the new and old dataset for training 
dataset_train = pd.concat([dataset_train, dataset_original], ignore_index=True)

# Count occurrences of each class
num_negatives = np.sum(np.array(dataset_train[target]) == 0)  # Number of class 0
num_positives = np.sum(np.array(dataset_train[target]) == 1)  # Number of class 1




#dataset_train = dataset_train.drop_duplicates()


df_train = dataset_train[features].copy()
df_test = dataset_test[features].copy()
df_val = dataset_original[features].copy()

y_train = np.array(dataset_train[target]).reshape(-1)
y_val = np.array(dataset_original[target]).reshape(-1)


# import matplotlib.pyplot as plt

plt.figure(figsize=[20,10])
plt.plot(df_train['sunshine'])
plt.show()


# Compute scale_pos_weight
scale_pos_weight = 1# num_negatives / num_positives

# weight dict
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
#class_weight_dict = {0: weights[0], 1: weights[1]}
class_weight_dict = {0:1, 1: 1}



import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=dataset_train, x='month', y='humidity', hue='rainfall', palette='viridis')
plt.title("Scatterplot: Cloud vs Sunshine (by Rainfall)")
plt.show()


import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
indices = np.arange(len(y_train))

xgb_auc_scores = []
xgb_test_preds = []
xgb_auc_scores_fold = []
fold = 1
for train_index, val_index in kf.split(indices):
    print(f"\n=== XGBoost Fold {fold} ===")
    
    # Split the training and validation sets
    X_train_fold = df_train.iloc[train_index].copy()
    X_val_fold   = df_train.iloc[val_index].copy()
    y_train_fold = y_train[train_index]
    y_val_fold   = y_train[val_index]
    
    # Initialize the XGBoost classifier with parameters similar to your CatBoost settings
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        colsample_bytree=0.6,
        random_state=42,
        reg_lambda=18,             
        reg_alpha=0.6, 
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'       # Use logloss for binary classification
    )
    
    # Train the model with early stopping on the validation set
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=180,
        verbose=False
    )

    xgb.plot_importance(xgb_model, importance_type='gain')
    plt.show()
    # Predict probabilities for the validation fold (probability for positive class)
    y_val_pred_fold = xgb_model.predict_proba(X_val_fold)[:, 1]
    auc_score_fold = roc_auc_score(y_val_fold, y_val_pred_fold)
    
    # Predict probabilities for the validation set (probability for positive class)
    y_val_pred = xgb_model.predict_proba(df_val)[:, 1]
    auc_score_val = roc_auc_score(y_val, y_val_pred)
    
    xgb_auc_scores_fold.append(auc_score_fold)
    xgb_auc_scores.append(auc_score_val)
    print(f"XGBoost Fold {fold} AUC fold: {auc_score_fold}")
    print(f"XGBoost Fold {fold} AUC Val: {auc_score_val}")

    # Predict on the test set for this fold
    y_test_pred = xgb_model.predict_proba(df_test)[:, 1]
    xgb_test_preds.append(y_test_pred)
    
    fold += 1

# Average test predictions and ROC-AUC over folds
final_test_predictions_xgb = np.mean(xgb_test_preds, axis=0)
avg_auc_score = np.mean(xgb_auc_scores)
print(f"\nAverage AUC over {n_splits} Val: {avg_auc_score}")
avg_auc_score = np.mean(xgb_auc_scores_fold)
print(f"\nAverage AUC over {n_splits} folds: {avg_auc_score}")



n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
indices = np.arange(len(y_train))


lgb_fold_metrics = []
lgb_test_preds = []
lgb_auc_scores = []

fold = 1
for train_index, val_index in kf.split(indices):
    print(f"\n=== LightGBM KM Fold {fold} ===")
    
    X_train_fold = df_train.iloc[train_index].copy()
    X_val_fold = df_train.iloc[val_index].copy()
    y_train_fold = y_train[train_index]
    y_val_fold = y_train[val_index]
    #sw_fold = sample_weights[train_index]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, )
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold,  reference=train_data)
    
    params = {
      "boosting_type": "rf",
        'n_estimators': 400,
        "metric": 'binary',
        'random_state': 777,
        'subsample': 0.2,
        'colsample_bytree': 0.35,
        "max_depth": 3,
        "learning_rate": 0.005,
        "verbose": -1,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "extra_trees":True,
        'num_leaves':16,
        'scale_pos_weight':scale_pos_weight,
    }
    
    lgb_model = lgb.train(params,
                          train_data,
                          num_boost_round=300,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=50)]
                          )
    
    y_val_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
    auc_score =  roc_auc_score(y_val_fold, y_val_pred)
    lgb_auc_scores.append(auc_score)
    print(f"LightGBM Fold {fold} AUC: {auc_score}")
    
    ax = lgb.plot_importance(lgb_model, max_num_features=20, importance_type='gain', figsize=(10, 6))
    plt.title("Feature Importance")
    plt.show()
    
    y_test_pred= lgb_model.predict(df_test, num_iteration=lgb_model.best_iteration)
    lgb_test_preds.append(y_test_pred)
    
    fold += 1

final_test_predictions_lgb = np.mean(lgb_test_preds, axis=0)

avg_auc_score = np.mean(lgb_auc_scores)
print(f"\nAverage AUC over {n_splits} folds: {avg_auc_score}")


import tensorflow as tf

def RainX(x_train_shape):

    # Input layer
    x_input = tf.keras.Input(shape=x_train_shape, name='x_input')    # Dense layers architecture
    x = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(l1=0.01))(x_input)
    x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)
    x = tf.keras.layers.Dense(150, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_3')(x)
    x = tf.keras.layers.Dense(80, activation='relu')(x)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    # Build and compile model
    model = tf.keras.Model(inputs=[x_input], outputs=output, name='ConCatModel')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])
    return model


import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Assume X and y are your features and labels (as numpy arrays)
input_shape = (df_train.shape[1],)  # e.g., (number_of_features,)

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
val_auc_scores = []
test_preds = []  # if you have a test set X_test

for train_index, val_index in kf.split(indices):
    print(f"\n=== MLP Fold {fold} ===")
    
    X_train_fold = df_train.iloc[train_index].copy()
    X_val_fold = df_train.iloc[val_index].copy()
    y_train_fold = y_train[train_index]
    y_val_fold = y_train[val_index]
    
    # Build a new instance of your model for this fold
    model = RainX(input_shape)
    
    # Set up early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=100,
        class_weight = class_weight_dict,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )
    
    
    # Predict on the validation set to calculate AUC (optional if not already provided by evaluate)
    y_val_pred = model.predict(X_val_fold).ravel()  # flatten predictions
    auc = roc_auc_score(y_val_fold, y_val_pred)
    print(f"Fold {fold} computed ROC AUC: {auc}")
    val_auc_scores.append(auc)
    
    #  predict and store test predictions
    y_test_pred = model.predict(df_test)
    test_preds.append(y_test_pred)
    
    fold += 1

avg_auc = np.mean(val_auc_scores)
print(f"\nAverage ROC AUC over {n_splits} folds: {avg_auc}")

# If you have test predictions from each fold, you could average them:
final_test_predictions_mlp = np.mean(test_preds, axis=0)



n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
val_auc_scores = []
test_preds = []  # if you have a test set X_test
test_preds_cnn = []

for train_index, val_index in kf.split(indices):
    print(f"\n=== CNN Fold {fold} ===")
    
    X_train_fold = df_train.iloc[train_index].copy()
    X_val_fold = df_train.iloc[val_index].copy()
    y_train_fold = y_train[train_index]
    y_val_fold = y_train[val_index]
    
    # Build a new instance of your model for this fold
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(X_train_fold.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
        
    tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    #tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)

    
    
    # Train the model
    history = model.fit(
    X_train_fold, y_train_fold, 
    epochs=200, 
    batch_size=256,
    class_weight = class_weight_dict,
    validation_data=(X_val_fold, y_val_fold),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
    
    
    # Predict on the validation set to calculate AUC (optional if not already provided by evaluate)
    y_val_pred = model.predict(X_val_fold).ravel()  # flatten predictions
    auc = roc_auc_score(y_val_fold, y_val_pred)
    print(f"Fold {fold} computed ROC AUC: {auc}")
    val_auc_scores.append(auc)
    
    #  predict and store test predictions
    y_test_pred = model.predict(df_test)
    test_preds_cnn.append(y_test_pred)
    
    fold += 1

avg_auc = np.mean(val_auc_scores)
print(f"\nAverage ROC AUC over {n_splits} folds: {avg_auc}")

# If you have test predictions from each fold, you could average them:
final_test_predictions_cnn = np.mean(test_preds_cnn, axis=0)



#final_test=(final_test_predictions_lgb+final_test_predictions_xgb+final_test_predictions_mlp[:,0])/3
#final_test = (2*final_test_predictions_cnn[:,0]+final_test_predictions_lgb)/3
final_test = (3*final_test_predictions_cnn[:,0]+2*final_test_predictions_mlp[:,0]+final_test_predictions_lgb+final_test_predictions_xgb)/7


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub["rainfall"] = final_test
sub =sub.fillna(sub.mean())
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head(30)

