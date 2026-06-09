def warn(*args, **kwargs):
    pass

import warnings 
warnings.warn = warn


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


print(train.shape, training_extra.shape,test.shape)


train.info()


train.describe()


train.id.is_unique


train.isnull().sum()


test.isnull().sum()


def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical column
            # Fill categorical missing values with NaN (you can leave this step as it is since they are already NaN)
            df[col] = df[col].fillna('NA')
        else:  # Continuous column
            # Fill continuous missing values with 0
            df[col] = df[col].fillna(0)
    return df

# Apply the function to the DataFrame
train = fill_missing(train)


fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].hist(train['Compartments'], color = 'blue', alpha = 0.7)
ax[0].set_title('Compartent Histogram')

ax[1].hist(train['Weight Capacity (kg)'], color = 'green', alpha = 0.7)
ax[1].set_title('Weight Capacity Histogram')

ax[2].hist(train['Price'], color = 'red', alpha = 0.7)
ax[2].set_title('Price Histogram')


# Identify categorical columns (excluding the target numeric column)
categorical_cols = train.select_dtypes(include=['object']).columns

# Create a figure with multiple rows (one per categorical column)
plt.figure(figsize=(6, 5 * len(categorical_cols)))  # Adjust height dynamically

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)  # One column, multiple rows
    sns.boxplot(x=train[col], y=train['Price'])
    plt.title(f'Boxplot of Price by {col}')
    plt.xticks(rotation=45)  # Rotate for better visibility if needed

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


corr_matrix = train.select_dtypes(include=['float64']).corr()
corr_matrix['Price'].sort_values(ascending=False)


sns.pairplot(train.select_dtypes(include=['float64']))
plt.show()


num = train.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(30,25))
sns.heatmap(num.corr(), annot = True, cmap = 'YlGnBu')
plt.show()



for col in train.select_dtypes(include=['object']).columns:
    print(train[col].value_counts())
    print('-'*40)


one_hot_columns = train.dtypes[train.dtypes == object].index
train = pd.get_dummies(train, columns = one_hot_columns, drop_first = True)


train.drop(['id'], axis = 1, inplace = True)


X = train.loc[:, train.columns != 'Price'].astype(np.float64)
y = train.loc[:,'Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


pf = PolynomialFeatures(degree = 2, include_bias = False)
X_train = pf.fit_transform(X_train)
X_test = pf.transform(X_test)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

np.sqrt(mean_squared_error(y_pred, y_test))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


model = Sequential()

model.add(Dense(64, input_dim = 27, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[rmse])


model.fit(X_train, y_train, epochs=100, batch_size=32)


model.save('my_trained_model.h5')


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train = pd.concat([data,data_extra], ignore_index = True)


def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical column
            # Fill categorical missing values with NaN (you can leave this step as it is since they are already NaN)
            df[col] = df[col].fillna('NA')
        else:  # Continuous column
            # Fill continuous missing values with 0
            df[col] = df[col].fillna(0)
    return df


train = fill_missing(train)


one_hot_columns = train.dtypes[train.dtypes == object].index
train = pd.get_dummies(train, columns = one_hot_columns, drop_first = True)


train.drop(['id'], axis = 1, inplace = True)


X = train.loc[:, train.columns != 'Price'].astype(np.float64)
y = train.loc[:,'Price']


ss = StandardScaler()
X = ss.fit_transform(X)


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')

# instantiate a distribution strategy
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


with tpu_strategy.scope():
    
    model = Sequential()

    model.add(Dense(64, input_dim = 27, activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[rmse])


model = Sequential()

model.add(Dense(64, input_dim = 27, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[rmse])


model = load_model('/kaggle/working/my_model.keras', custom_objects={'rmse': rmse})
#model = load_model('/kaggle/working/my_trained_model.h5')
#model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[rmse])


model.save('my_model_backup.keras')
!rm -rf /kaggle/working/my_model.keras


# Check devices
print(tf.config.list_physical_devices())

# Optional (force to use GPU if available)
if tf.config.list_physical_devices('GPU'):
    print("✅ Using GPU!")
else:
    print("❌ GPU not available - check your Kaggle settings")


model.fit(X, y, epochs=5, batch_size=64)
model.save('my_model.keras')


model.save('my_model.keras')


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test = fill_missing(test)


one_hot_columns = test.dtypes[test.dtypes == object].index
test = pd.get_dummies(test, columns = one_hot_columns, drop_first = True)


id = test['id']
test.drop(['id'], axis = 1, inplace = True)


test = ss.transform(test)


prediction = model.predict(test)


prediction = prediction.reshape(-1,1)


id = pd.DataFrame(id)


id['Price'] = prediction


!rm -rf /kaggle/working/submission.csv


id.to_csv('submission.csv', index = False)







