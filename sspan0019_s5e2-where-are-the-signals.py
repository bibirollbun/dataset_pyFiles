import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


train       = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
test        = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

train = pd.concat([train, train_extra], axis=0, ignore_index=True)


print(train.shape, test.shape)


train.info()


CAT_COLS = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
NUM_COLS = ['Compartments', 'Weight Capacity (kg)']


msno.bar(train)


print('Overall average price:', train['Price'].mean())
print()

missing_value_avg_price = train[train.isnull().any(axis=1)].groupby(train.isnull().idxmax(axis=1))['Price'].mean()
print(missing_value_avg_price)


for col in CAT_COLS:
    print(f'Unique values for {col}:')
    print(train[col].unique())
    print()


for col in CAT_COLS:
    avg_price_per_value = train.groupby(col)['Price'].mean()
    print(f'Average price for each unique value in {col}:')
    print(avg_price_per_value)
    print()


plt.figure(figsize=(16, 4))

for i, col in enumerate(NUM_COLS + ['Price']):
    plt.subplot(1, len(NUM_COLS) + 1, i + 1)
    sns.histplot(train[col], bins=10)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


corr_matrix = train[NUM_COLS + ['Price']].corr()

plt.figure(figsize=(16, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Numerical Features')
plt.show()


train[CAT_COLS] = train[CAT_COLS].fillna('unknown')
test[CAT_COLS] = test[CAT_COLS].fillna('unknown')


train = pd.get_dummies(train, columns=CAT_COLS)
test  = pd.get_dummies(test, columns=CAT_COLS)


bins = [0, 10, 20, 30, float('inf')]
labels = [1, 2, 3, 4]

train['Weight Category'] = pd.cut(train['Weight Capacity (kg)'], bins=bins, labels=labels, right=False).cat.add_categories([-1]).fillna(-1)
test['Weight Category'] = pd.cut(test['Weight Capacity (kg)'], bins=bins, labels=labels, right=False).cat.add_categories([-1]).fillna(-1)

train = train.drop(columns=['Weight Capacity (kg)'])
test  = test.drop(columns=['Weight Capacity (kg)'])


train['Compartments'] = train['Compartments'].fillna(-1)
test['Compartments'] = test['Compartments'].fillna(-1)


train = train.astype(float)
test = test.astype(float)


COLS = test.columns.tolist()

encoder = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
encoder.fit(train[COLS], train['Price'])
train[COLS] = encoder.transform(train[COLS])
test[COLS] = encoder.transform(test[COLS])


scaler = StandardScaler()
train[COLS] = scaler.fit_transform(train[COLS])
test[COLS] = scaler.transform(test[COLS])


train.head()


predictors = train.drop(columns='Price')
target = train['Price']

train_predictors, eval_predictors, train_target, eval_target = train_test_split(predictors, target, test_size=0.2, random_state=42)


sgd_model = SGDRegressor()
sgd_model.fit(train_predictors, train_target)


eval_predictions = sgd_model.predict(eval_predictors)
eval_mse = mean_squared_error(eval_target, eval_predictions)
eval_rmse = np.sqrt(eval_mse)
print(f'SGD RMSE: {eval_rmse:.4f}')


eval_mean = eval_target.mean()
eval_predictions = np.full_like(eval_target, fill_value=eval_mean)
eval_mse = mean_squared_error(eval_target, eval_predictions)
eval_rmse = np.sqrt(eval_mse)
print(f'Mean RMSE: {eval_rmse:.4f}')



predictors = train.drop(columns=['Price'])
target = train['Price']


nn_model = Sequential()
nn_model.add(Input(shape=(predictors.shape[1],)))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(1, activation='linear'))

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
nn_model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)

train_history = nn_model.fit(predictors, target, 
                    validation_split=0.2,
                    epochs=50,
                    batch_size=1024, 
                    callbacks=[early_stopping, reduce_lr])


plt.figure(figsize=(16, 9))
plt.plot(train_history.history['loss'][1:], label='train loss')
plt.plot(train_history.history['val_loss'][1:], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


test_predictions = nn_model.predict(test)
test['Price'] = test_predictions
test[['Price']].to_csv('s5e2-submission.csv', index=True)

