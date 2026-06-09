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



target_col = "responder_6"
necessary_cols = [target_col, 'weight']
feat_clear_categ = ["feature_09", "feature_10", "feature_11"]
feature_categ = feat_clear_categ + ['symbol_id', 'time_id']
feature_cols = [f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11, 61]]
responder_cols = [f"responder_{idx}_lag_1" for idx in range(9)] 
feature_cont = feature_cols + responder_cols
dataset_cols = feature_cont + necessary_cols + feature_categ
std_feature = [i for i in feature_cont]

batch_size = 8192
n_cont_features = len(feature_cont)
n_cat_features = len(feature_categ)
n_classes = None
cat_cardinalities = [23, 10, 32, 40, 969]


import pandas as pd


path = f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=1/part-0.parquet"
df = pd.read_parquet(path)



df.head()


df[feature_cols]


df.shape



target_col = "responder_6"
# feature_cols = ["symbol_id", "time_id"] + [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
feature_cols = ["symbol_id", "time_id"]+[f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}" for idx in range(9)]


from sklearn.model_selection import train_test_split

df = pd.read_parquet(path)
df=df.fillna(method='ffill').fillna(0)


# Split the data
x_train, x_val, y_train, y_val = train_test_split(df[feature_cols], df['responder_6'], test_size=0.2, shuffle=False)


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block_1d(input_layer, filters, kernel_size=3, strides=1):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding="same", activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding="same")(input_layer)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet_variable(input_features, num_classes):
    inputs = layers.Input(shape=(None, input_features))  
    
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)
    x = resnet_block_1d(x, filters=64)
    x = resnet_block_1d(x, filters=128, strides=2)
    x = resnet_block_1d(x, filters=256, strides=2)
    x = resnet_block_1d(x, filters=512, strides=2)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax" if num_classes > 1 else "linear")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model






input_shape = x_train.shape[1] # (time_steps, features), modify as needed
num_classes = 1  # For classification (set to 1 for regression)

model = build_resnet_variable(input_shape, num_classes)
model.compile(optimizer="adam", 
              loss="mse", 
              metrics=["mae"])

model.summary()





y_train.shape


y_train = y_train.to_numpy()

# Reshape to (samples, time_steps, features) for Conv1D
# Assuming each row is a time series with 100 time steps and 1 feature



import numpy as np
y_train = np.array(y_train, dtype=np.float32)



y_train.shape



x_train=x_train.to_numpy()


x_train.shape




x_train = x_train.reshape(x_train.shape[2], x_train.shape[1],x_train.shape[0])  # Reshape to (500, 1)






y_train.reshape(0,1,y_train[0])


x_train.shape



model.fit(x_train, y_train, epochs=2, batch_size=32)


import joblib

joblib.dump(model, 'ResNetModel.pkl')



x_val.shape[0]


x_val = x_val.to_numpy()



x_val = x_val.reshape(x_val.shape[0],1, x_val.shape[1])



x_val.shape


from sklearn.metrics import r2_score

predictions = model.predict(x_val)
print(predictions)
r2_score_test = r2_score(y_val, predictions)
print(f"R2 Score on Test Partition: {r2_score_test}")


tt=predictions.reshape(-1)


tt.shape


import os
import polars as pl
import kaggle_evaluation.jane_street_inference_server


# lags_ : pl.DataFrame | None = None

# # Replace this function with your inference code.
# # You can return either a Pandas or Polars dataframe, though Polars is recommended.
# # Each batch of predictions (except the very first) must be returned within 10 minutes of the batch features being provided.
# def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
#     """Make a prediction."""
#     # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
#     # Use them as extra features, if you like.
#     global lags_
#     if lags is not None:
#         lags_ = lags

#     predictions = test.select(
#         'row_id',
#         pl.lit(0.0).alias('responder_6'),
#     )
    
#     feature_names = [f"feature_{i:02d}" for i in range(79)]
#     feat = test[feature_names].to_numpy()
    
#     pred = [model.predict(feat) for model in models_1]
#     pred = np.mean(pred, axis=0)
#     # print("Pred =", pred)
#     # predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
#     symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

#     if not lags is None:
#         lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
#         test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
#     else:
#         test = test.with_columns(
#             ( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) )
#         )
    
#     preds = np.zeros((test.shape[0],))
#     preds += xgb_model.predict(test[xgb_feature_cols].to_pandas()) / 2
    
#     test_input = test[CONFIG.feature_cols].to_pandas()
#     test_input = test_input.fillna(method = 'ffill').fillna(0)
#     test_input = torch.FloatTensor(test_input.values).to("cuda:0")
#     with torch.no_grad():
#         for i, nn_model in enumerate(tqdm(models)):
#             nn_model.eval()
#             preds += nn_model(test_input).cpu().numpy() / 10
            
#     # print(f"predict> preds.shape =", preds.shape)
    
#     # print(pred.shape)
#     final_pred = (preds + pred) / 2
#     predictions = \
#     test.select('row_id').\
#     with_columns(
#         pl.Series(
#             name   = 'responder_6', 
#             values = np.clip(final_pred, a_min = -5, a_max = 5),
#             dtype  = pl.Float64,
#         )
#     )
#     # print(predictions)
#     # The predict function must return a DataFrame
#     assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
#     # with columns 'row_id', 'responer_6'
#     assert list(predictions.columns) == ['row_id', 'responder_6']
#     # and as many rows as the test data.
#     assert len(predictions) == len(test)

#     return predictions


df


lags_: pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    feature_names = [f"feature_{i:02d}" for i in range(90)]
    feat = test[feature_names].to_numpy()
    feat = feat.reshape(feat.shape[0], 1, feat.shape[1])

    pred = model.predict(feat)
    pred=pred.reshape(-1)
    print(pred.shape, pred.min(), pred.max())
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))

    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions



inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )


feature_names = df.columns



feature_names


lags_ : pl.DataFrame | None = None
feature_names = df.columns

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 10 minutes of the batch features being provided.
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    
    feat = test[feature_names].to_numpy()
    feat = feat.reshape(feat.shape[0], 1, feat.shape[1])
    pred = model.predict(feat)
    pred=pred.reshape(-1)

    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions




