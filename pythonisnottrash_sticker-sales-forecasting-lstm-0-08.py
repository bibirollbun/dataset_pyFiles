import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


print(tf.__version__)


# check for GPU availability
print("Is GPU available?", tf.config.list_physical_devices('GPU'))


df = pd.read_csv('train.csv',parse_dates=[1], index_col=1)
df.head()


test_df = pd.read_csv('test.csv',parse_dates=[1], index_col=1)
test_df.head()


df.isna().sum()


# filling with previos value
df['num_sold'] = df.groupby(['country', 'date'])['num_sold'].bfill()


# train encoding
# county enc
encoder_c = OneHotEncoder(sparse_output=False)
c_encoded = encoder_c.fit_transform(df[['country']])
# store enc
encoder_s = OneHotEncoder(sparse_output=False)
s_encoded = encoder_s.fit_transform(df[['store']])

#product enc
encoder_p = OneHotEncoder(sparse_output=False)
p_encoded = encoder_p.fit_transform(df[['product']])


# test encoding
t_encoder_c = OneHotEncoder(sparse_output=False)
t_c_encoded = t_encoder_c.fit_transform(test_df[['country']])
# store enc
t_encoder_s = OneHotEncoder(sparse_output=False)
t_s_encoded = t_encoder_s.fit_transform(test_df[['store']])

#product enc
t_encoder_p = OneHotEncoder(sparse_output=False)
t_p_encoded = t_encoder_p.fit_transform(test_df[['product']])


# create df from np
country_encoded = pd.DataFrame(
    c_encoded, 
    columns=encoder_c.get_feature_names_out(['country']), 
    index=df.index  # Preserve original DataFrame index
)

store_encoded = pd.DataFrame(
    s_encoded, 
    columns=encoder_s.get_feature_names_out(['store']), 
    index=df.index)

product_encoded = pd.DataFrame(
    p_encoded, 
    columns=encoder_p.get_feature_names_out(['product']), 
    index=df.index)


# creating dfs with encoded values
t_country_encoded = pd.DataFrame(
    t_c_encoded, 
    columns=t_encoder_c.get_feature_names_out(['country']), 
    index=test_df.index  # Preserve original DataFrame index
)

t_store_encoded = pd.DataFrame(
    t_s_encoded, 
    columns=t_encoder_s.get_feature_names_out(['store']), 
    index=test_df.index)

t_product_encoded = pd.DataFrame(
    t_p_encoded, 
    columns=t_encoder_p.get_feature_names_out(['product']), 
    index=test_df.index)


df = pd.concat([df, country_encoded], axis=1).drop(columns='country')
df = pd.concat([df, store_encoded], axis=1).drop(columns='store')
df = pd.concat([df, product_encoded], axis=1).drop(columns='product')


test_df = pd.concat([test_df, t_country_encoded], axis=1).drop(columns='country')
test_df = pd.concat([test_df, t_store_encoded], axis=1).drop(columns='store')
test_df = pd.concat([test_df, t_product_encoded], axis=1).drop(columns='product')
test_df.head()


# calendar features
df['year'] = pd.DatetimeIndex(df.index).year
df['month'] = pd.DatetimeIndex(df.index).month
df['week'] = df.index.dayofweek
df['day'] = pd.DatetimeIndex(df.index).day

test_df['year'] = pd.DatetimeIndex(test_df.index).year
test_df['month'] = pd.DatetimeIndex(test_df.index).month
test_df['week'] = test_df.index.dayofweek
test_df['day'] = pd.DatetimeIndex(test_df.index).day


# sin/cos

# month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# week
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 7)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 7)
# day 
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

#test month
test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] / 12)
test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] / 12)
#test week
test_df['week_sin'] = np.sin(2 * np.pi * test_df['week'] / 7)
test_df['week_cos'] = np.cos(2 * np.pi * test_df['week'] / 7)
# day 
test_df['day_sin'] = np.sin(2 * np.pi * test_df['day'] / 31)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day'] / 31)


df.describe()


df.shape


test_split=round(len(df)*0.20)
test_split


dftrain = df[:-46026]
dftest = df[-46026:]
print(dftrain.shape)
print(dftest.shape)


# scale all features not including binary ones
scaler = MinMaxScaler(feature_range=(0,1))
scal_list = ['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos']
notbin_train_scaled = scaler.fit_transform(dftrain[scal_list])
notbin_test_scaled=scaler.transform(dftest[scal_list])

# scale test_df
scal_list = ['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos']
scaler_td = MinMaxScaler(feature_range=(0,1))
tdf_scaled = scaler_td.fit_transform(test_df[scal_list])


tr_sold_cbrt = dftrain[['num_sold']] ** (1/3)
te_sold_cbrt = dftest[['num_sold']] ** (1/3)


te_sold_cbrt.head(4)


# average daily cbrt values 
tr_daily_mean = tr_sold_cbrt.groupby('date', as_index=False)['num_sold'].mean()


# Plot a histogram to check the distribution of the data
# kde=True adds a Kernel Density Estimation (KDE) curve to show the smoothed distribution
plt.figure(figsize=(10,5))
sns.histplot(tr_daily_mean, kde=True)


# сreating datasets (pandas dfs) from scaled arrays (numpy)
tr_notbin_scaled = pd.DataFrame(
    notbin_train_scaled, 
    columns=['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'],
    index=dftrain.index)

te_notbin_scaled = pd.DataFrame(
    notbin_test_scaled, 
    columns=['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'],
    index=dftest.index)


tr_sold_scaled = pd.DataFrame(
    tr_sold_cbrt, 
    columns=['num_sold'], 
    index=dftrain.index)

te_sold_scaled = pd.DataFrame(
    te_sold_cbrt, 
    columns=['num_sold'], 
    index=dftest.index)
# drop old unscaled values 
dftrain = dftrain.drop(['num_sold','day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'], axis=1)
dftest = dftest.drop(['num_sold','day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'], axis=1)


# same with test_df
fin_test_scaled = pd.DataFrame(
    tdf_scaled, 
    columns=['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'],
    index=test_df.index)

test_df = test_df.drop(['day','week','month','year','month_sin','month_cos','week_sin','week_cos','day_sin','day_cos'], axis=1)


dftrain.head(2)


dftrain_s = pd.concat([dftrain, tr_notbin_scaled], axis=1)
dftest_s = pd.concat([dftest, te_notbin_scaled], axis=1)

dftrain_s = pd.concat([dftrain_s, tr_sold_scaled], axis=1)
dftest_s = pd.concat([dftest_s, te_sold_scaled], axis=1)


test_df_s = pd.concat([test_df, fin_test_scaled], axis=1)


dftrain_s.head(2)


# drop ids
dftrain_s = dftrain_s.drop(['id'], axis=1)
dftest_s = dftest_s.drop(['id'], axis=1)
test_df_s = test_df_s.drop(['id'], axis=1)


dftrain_s.head(2)


test_df_s.head(2)


# num_sold to first pos
column_to_move = 'num_sold'

# move the column to the first position
columns = [column_to_move] + [col for col in dftrain_s.columns if col != column_to_move]
dftrain_s = dftrain_s[columns]

columns_t = [column_to_move] + [col for col in dftest_s.columns if col != column_to_move]
dftest_s= dftest_s[columns_t]
dftest_s.head(2)


test_df_s.head(2)


dftrain_s = dftrain_s.values
dftest_s = dftest_s.values

fin_test_df = test_df_s.values


# sliding windows function
def sliding_w(dataset,n_past):
    X = []
    Y = []
    for i in range(n_past, len(dataset)):
            X.append(dataset[i - n_past:i, 1:dataset.shape[1]]) # "1:dataset.shape[1]]" defines X from the next function after the target function
            Y.append(dataset[i,0]) # target col
    return np.array(X),np.array(Y)

trainX,trainY=sliding_w(dftrain_s,1) # apply the sliding window function to the training dataset (1 step back for prediction)
testX,testY=sliding_w(dftest_s,1) # apply the sliding window function to the test dataset (1 step back for prediction)


trainX.shape


testX.shape


timestemps = trainX.shape[1] #20 timestemps
features = trainX.shape[2] #16 features
#we predict 1 feature to the future

# creating a model
model = Sequential()

#add 1st LSTM layer with 16 neurons
model.add(LSTM(units=16, activation='relu', return_sequences=True, input_shape=(timestemps, features)))
#model.add(Dropout(0.2))
#model.add(BatchNormalization(axis=1))
#one more time
model.add(LSTM(units=16, activation='relu', return_sequences=True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization(axis=1))

model.add(LSTM(units=16, activation='relu', return_sequences=False))
#model.add(Dropout(0.1))
#model.add(BatchNormalization(axis=1))

# layer for output
model.add(Dense(1, activation='relu'))
#model.add(Dense(1))
#compelation
model.compile(optimizer=Adam(learning_rate=0.01), loss='mae',metrics=['mae'])#'mae'])


checkpoint_filepath = 'models/checkpoint.model.keras' # where to save the best model

callbacks = [ EarlyStopping(
    monitor='val_mae', # monitor validation mae
    verbose=1,
    patience=30, # how many epochs to wait to stop learning
    mode='min', # the lowest error
    restore_best_weights=True, # restore the best weights
),
    ModelCheckpoint(
    filepath=checkpoint_filepath,  # where to save the best model
    monitor='val_mae',       # monitor validation mae
    save_best_only=True,# save only when improving
    mode='min', # the lowest error
    verbose=1
    )
]


history = model.fit(trainX,trainY,
                    validation_data=(testX, testY), # Data used for validation during training (to monitor overfitting)
                    epochs=500, # num of epochs
                    batch_size=512, # Size of each batch during training (number of samples per update)
                    callbacks=[callbacks], # List of callback functions to use (for monitoring, early stopping and saving the best model)
                    verbose=1) # Level of verbosity for logging progress during training (1 means showing progress bar)


# load the best model
model = load_model(checkpoint_filepath)


plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# test loss
test_loss = model.evaluate(testX, testY)
print("Test Loss:", test_loss)


predictions = model.predict(testX)
predictions


print(testY.shape)
print(predictions.shape)


# reverse transformation for cbrt
y_true = testY ** 3
y_pred = predictions ** 3


y_pred


y_true


mape = mean_absolute_percentage_error(y_true, y_pred)

print(f'MAPE: {mape}%')


root_mean_squared_error(y_true, y_pred)


def sliding_w(dataset,n_past):
    X = []
    for i in range(n_past, len(dataset)):
            X.append(dataset[i - n_past:i, 0:dataset.shape[1]]) # includes all features
    return np.array(X)

finalX=sliding_w(fin_test_df,1)


final_pred = model.predict(finalX)


final_pred


# reverse transformation for cbrt
fin_predictions = final_pred ** 3


fin_predictions


subm_test = pd.read_csv('test.csv')
subm_test.shape


fin_predictions.shape


subm_test.shape


# fill in any missing lines caused by the sliding windows
missing_rows = len(subm_test) - len(fin_predictions)
mean_prediction = np.mean(fin_predictions)
padding = np.full((missing_rows, 1), mean_prediction)
padded_predictions = np.vstack([padding, fin_predictions])


padded_predictions.shape


submission_df = subm_test[['id']].copy()
submission_df['num_sold'] = padded_predictions
submission_df


# save submission
submission_df.to_csv('submission.csv', index=False)

