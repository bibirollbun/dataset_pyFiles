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


train_data = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_data =  pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_data =  pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")


X_train = train_data[[col for col in train_data.columns if col != 'Lap_Time_Seconds']]
y_train = train_data['Lap_Time_Seconds']

X_val = val_data[[col for col in val_data.columns if col != 'Lap_Time_Seconds']]
y_val = val_data['Lap_Time_Seconds']

X_test = test_data

print(f"\n{X_train.info()}\n")
#Columns
print(f"Number of Features: {len(X_train.columns)}")
print(f"List of Features: {X_train.columns}")


print(f"Number of Features: {len(X_val.columns)}")
print(f"List of Features: {X_val.columns}")




numerical_columns = []
categorical_columns = []
for i, dtype in enumerate(X_train.dtypes):
    if dtype == 'object':
        categorical_columns.append(X_train.dtypes.index[i])
    else:
        numerical_columns.append(X_train.dtypes.index[i])

print(f"Categorical Columns: {categorical_columns}\n---------------------") #Needs to be encoded
print(f"Numerical Columns: {numerical_columns}")


X_train[numerical_columns].describe()
#No need to scale the data.


for col in categorical_columns:
    print(f"Number of unique values in {col}: {X_train[col].nunique()}")
#Dropping rider_name,team_name,bike_name.


print(f"Number of null values in each column: {X_train.isnull().sum()}")


#Columns that can be dropped since they are not relevant to performance/ lap timing.
columns_to_be_dropped = ['shortname','circuit_name','rider_name','team_name','bike_name','Unique ID', 'Penalty']
X_train = X_train.drop(columns=columns_to_be_dropped, errors='ignore')
X_val  = X_val.drop(columns=columns_to_be_dropped, errors='ignore')
X_test = X_test.drop(columns=columns_to_be_dropped, errors='ignore')



len(X_val.columns), len(X_train.columns), len(X_test.columns)  #making sure there are no errors.


print(f"Final Columns: {len(X_train.columns)}")

#Encoding
one_hot_features = ['category_x', 'Tire_Compound_Front', 'Tire_Compound_Rear','Session']
ordinal_features = ['Track_Condition', 'weather', 'track']

for col in ordinal_features:
    print(X_train[col].unique())


numerical_features = [col for col in X_train.columns if col not in categorical_columns]
numerical_features


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


track_condition = ['Dry', 'Wet']
Weather = X_train['weather'].unique().tolist()
Track = ['Dry','Wet']

ordinal_encoder = OrdinalEncoder(categories=[track_condition, Weather, Track])

preprocessor = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(handle_unknown='ignore'), one_hot_features),
    ('ord', ordinal_encoder, ordinal_features),
    ('scaler',StandardScaler(), numerical_features)
]
)

model_xgb = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=2000,
        learning_rate=1,
        max_depth=9,
        random_state=27
    ))
])

model_xgb.fit(X_train,y_train)
val_preds = model_xgb.predict(X_val)
val_score = mean_squared_error(y_val, val_preds)
print(f"Validation Score of our model: {val_score}")




#1.498543550897384e-08


test_preds = model_xgb.predict(X_test)


print(type(test_preds))


test_pred_df = pd.DataFrame(test_preds, columns=['Lap_Time_Seconds'],index=test_data['Unique ID'])
test_pred_df.index.name = 'Unique ID'
test_pred_df.to_csv('/kaggle/working/submission.csv')


!pip install lightgbm


from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

preprocessor = ColumnTransformer(transformers=[
    ('oh', OneHotEncoder(handle_unknown='ignore'), one_hot_features),
    ('ord', ordinal_encoder, ordinal_features),
    ('scaler',StandardScaler(), numerical_features)
]
)


model_lgbm = Pipeline(steps=[
    ('preprocess', preprocessor), 
    ('regressor', LGBMRegressor(
        n_estimators=1000,
        learning_rate=1, 
        num_leaves=31,     
        random_state=27
    ))
])


model_lgbm.fit(X_train, y_train)


val_preds_lgbm = model_lgbm.predict(X_val)
val_score_lgbm = mean_squared_error(y_val, val_preds_lgbm)
print(f"Validation Score of LightGBM model: {val_score_lgbm}")


# #Deep NN
# import tensorflow as tf
# from tensorflow.keras import layers, models

# X_train_processed = preprocessor.fit_transform(X_train)
# X_val_processed = preprocessor.transform(X_val)

# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X_train_processed.shape[1],)),
    
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.3),
    
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.3),

#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss='mse',
#     metrics=[tf.keras.metrics.RootMeanSquaredError()]
# )

# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss', 
#         patience=5, 
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath='best_model.keras',
#         monitor='val_loss',
#         save_best_only=True
#     )
# ]

# history = model.fit(
#     X_train_processed, y_train,
#     validation_data=(X_val_processed, y_val),
#     epochs=100,
#     batch_size=1024,
#     callbacks=callbacks,
#     verbose=1
# )

# # Step 5: Evaluate on validation
# val_loss, val_rmse = model.evaluate(X_val_processed, y_val)
# print(f"Validation RMSE: {val_rmse}")

