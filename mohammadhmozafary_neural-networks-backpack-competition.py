# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras import regularizers
import keras_tuner as kt
from sklearn.decomposition import PCA
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train1=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train2=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


submission=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_set = pd.concat([train1, train2], axis=0, ignore_index=True)


train_set.info()


print(train1.shape,train2.shape,test.shape,train_set.shape )


test.isna().sum()/len(test)*100



def handle_missing_values(df):
    df=df.dropna()
    # Identify rows with missing values before imputation (optional)
    
    return df




def impute(df):
    # Separate categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    
    # Impute categorical features with the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Impute numerical features with the mean value
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Identify rows with missing values before imputation (optional)
    
    return df



def encoding(df,ordinals,convertables,encodables):
    df[convertables]=df[convertables].replace({'Yes':1,'No':0})
    df[ordinals]=df[ordinals].replace({'Small':1,'Medium':2,'Large':3})
    encoded = pd.get_dummies(df,columns=encodables,dtype=int)
    return encoded


def handle_outliers(df, columns):
    """
    This function removes outliers based on the IQR (Interquartile Range) method for specific columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to clean.
    columns (list): The list of column names on which to apply the outlier removal.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame with outliers removed from the specified columns.
    """
    # Loop through each specified column
    for col in columns:
        # Calculate Q1, Q3, and IQR for the current column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Filter out outliers in the current column
        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    
    return df



def scale_data(df):
    scaler=MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled


def feature_selection(df , pca):
    reduced=pca.transform(df)
    return reduced


def read_data(df,lower_index,upper_index):
    return df[lower_index:upper_index+1]


# Main PipeLine
def preprocessing_pipeline(train,test):
    testId=test['id']
    test=test.drop('id',axis=1)
    train=train.drop('id',axis=1)
    train=train.drop_duplicates()
    test=test.drop_duplicates()
    
    train=impute(train)
    test=impute(test)
    
    columns_to_convert=['Waterproof','Laptop Compartment']
    ordinals=['Size']
    columns_to_encode=['Brand','Material','Style','Color']
    train=encoding(train,ordinals,columns_to_convert,columns_to_encode)
    test=encoding(test,ordinals,columns_to_convert,columns_to_encode)
    
    
    iqrCheck_cols=['Compartments','Weight Capacity (kg)']
    train=handle_outliers(train,iqrCheck_cols)
    test=handle_outliers(test,iqrCheck_cols)
    
    
    
    target_column = 'Price'  
    X_train = train.drop(columns=[target_column])  # Input features
    y_train = train[target_column] 
    X_train=scale_data(X_train)
    X_test=scale_data(test)
    
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_train = pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    n_components=X_train.shape[1]
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    X_train = pd.DataFrame(X_train, columns=pca_columns)
    X_test=pd.DataFrame(X_test,columns=pca_columns)
    X_test['id']=testId
    return X_train,y_train,X_test
    
    
    




X_train_prep,y_train_prep,X_test_prep=preprocessing_pipeline(train_set,test)


X_test_prep.shape


X_train_prep.shape


X_test_prep.head()



# Assume X and y are your preprocessed NumPy arrays
X_tensor = tf.convert_to_tensor(X_train_prep, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_train_prep, dtype=tf.float32)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

# Shuffle with a buffer size (not the entire dataset at once)
buffer_size = 100000  # Pick a reasonable buffer size to avoid memory issues
dataset = dataset.shuffle(buffer_size=buffer_size)

# Split sizes
train_size = int(0.7 * len(X_train_prep))
val_size = int(0.15 * len(X_train_prep))
test_size = len(X_train_prep) - train_size - val_size

# Split the dataset
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

# Batch the datasets (important for training efficiency)
batch_size = 256
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Prefetching for performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Print dataset details
print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")



# Define a simple neural network
model = Sequential([
    
    Dense(64, input_shape=(X_train_prep.shape[1],)),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.3),
    BatchNormalization(),
    
    # Dense(128),
    # Dropout(0.3),
    # LeakyReLU(negative_slope=0.1),
    # BatchNormalization(),
    
    Dense(32,kernel_regularizer=regularizers.l2(0.005)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    
    Dense(16, kernel_regularizer=regularizers.l2(0.005)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    Dense(16, kernel_regularizer=regularizers.l2(0.005)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),

    Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='mse',  # MSE as the loss function
    metrics=[tf.keras.metrics.RootMeanSquaredError()]  # RMSE as the evaluation metric
)


# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Adjust epochs as needed
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Evaluate on test set

test_loss, test_rmse = model.evaluate(test_dataset)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test RMSE: {test_rmse}")

plt.figure(figsize=(12, 6))

# Plot RMSE
plt.subplot(1, 2, 1)
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
plt.title('Root Mean Squared Error (RMSE) Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

# Plot Loss (MSE)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title('Loss (MSE) Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.tight_layout()
plt.show()



predictions = model.predict(X_test_prep.iloc[:,:-1])



predictions


submission.head()


submission['Price']=predictions


submission.head()


submission.to_csv('submission.csv')




