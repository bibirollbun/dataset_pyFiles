# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import tensorflow as tf
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # check the number of GPU's available


warnings.filterwarnings("ignore") # filter out warnings


SEED = 42  # Choose a random seed and set it for python, np, tf, environment

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()

SUBMISSION = True # set to true when submitting to reduce processing time


print("Loading data....")


# Load the train and test datasets as df
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col = "id")


if not SUBMISSION:
    # Show basic df info
    train.info()


# Define categorical en numerical columns
l_num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
l_cat_cols = ["job", "marital", "month", "education", "default", "housing", "loan", "contact", "poutcome"]


if not SUBMISSION:
    # Check unique values present in categorical columns in train and test
    for col in l_cat_cols:
        print("train:", col, train[col].unique())
        print("test:", col, test[col].unique())
        print("-------")


if not SUBMISSION:
    # Explore class label distribution
    print(train[['y']].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

if not SUBMISSION:
    # Create KDE plots for each numerical column, separated by binary label 'y'
    # https://www.geeksforgeeks.org/data-visualization/kde-plot-visualization-with-pandas-and-seaborn/
    for col in l_num_cols:
        plt.figure(figsize=(5, 3))
        sns.kdeplot(data=train, x=col, hue='y', common_norm=False)
        plt.title(f'KDE Plot of {col} by Label y')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if not SUBMISSION:
    for col in l_cat_cols:
        # Create a count plot with categories on x-axis and count on y-axis, split by class label 'y'
        category_counts = train.groupby([col, 'y']).size().unstack(fill_value=0)
        plt.figure(figsize=(5, 3))
        category_counts.plot(kind='bar', stacked=True)
        plt.title(f'Plot of counts of {col} by Label y')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title='Class Label')
        plt.tight_layout()
        plt.show()


print("Preprocessing....")


from sklearn.preprocessing import OneHotEncoder

# One-hot encode our categorical features to make them numerical
encoder = OneHotEncoder(sparse_output=False)
train_cat_encoded = encoder.fit_transform(train[l_cat_cols]) # fit and transform the train set
test_cat_encoded = encoder.transform(test[l_cat_cols]) # transform the test set with the fitted encoder

# Convert to DataFrames with column names
encoded_columns = encoder.get_feature_names_out(l_cat_cols) # collect the feature names created by the encoder
train_cat_df = pd.DataFrame(train_cat_encoded, columns=encoded_columns, index=train.index)
test_cat_df = pd.DataFrame(test_cat_encoded, columns=encoded_columns, index=test.index)

# Drop original categorical columns and concatenate encoded ones
train = pd.concat([train.drop(columns=l_cat_cols), train_cat_df], axis=1)
test = pd.concat([test.drop(columns=l_cat_cols), test_cat_df], axis=1)


if not SUBMISSION:
    # Display train in current state
    train.info()


X = train.drop(["y"], axis=1) # Feature matrix
y = train["y"] # Target labels


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit a scaler and scale train data and convert to DataFrame
X = pd.DataFrame(scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
                )

# Scale test data with fitted scaler and convert to DataFrame
test = pd.DataFrame(scaler.transform(test),
                    columns=test.columns,
                    index=test.index
                    )


from sklearn.utils.class_weight import compute_class_weight

# Compute class weights which might come in handy for our model to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

class_weight_dict = dict(enumerate(class_weights))
class_weight_dict # Display computed class weights


print("Training model....")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Define our neural network model. The model is sequential meaning that 
# our inputs pass trough the network in a sequential manner, layer after layer.
# The model starts out very wide with a lot of neurons (256) in a single dense layer.
# With each dense layer, the model becomes narrower and narrower and eventually 
# outputs a single value between 0 and 1, which will be the prediction given one row of input features.

model = Sequential([
    Dense(256, activation='relu', input_dim=X.shape[1]), 
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2), 

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

# Compile our neural network
model.compile(optimizer=Adam(learning_rate=1e-2),
              loss='binary_crossentropy',
              metrics=['AUC'],
              )

# Define criteria for early stopping during training. 
# This will prevent the model from overfitting to the training data.
# The training will stop when the validation loss has not improved for 10 epochs
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=10,
                                                  verbose=1,
                                                  mode='auto',
                                                  restore_best_weights=True, # Restores the weights of the best epoch
                                                  start_from_epoch=20
                                                 )

# Fit our model on the data. Using a validation split of 0.2, the model will use 
# 80% of the data for training and the leftover for validation during the fitting process.
# The number of epochs determines the number of times the model will see our training data 
# during the fitting process. The batch size determines the size of each batch that gets 
# passed to our model in one training instance. Lastly, we pass our class_weights_dict so the model
# can take the class imbalance into account.
model.fit(X,
          y,
          validation_split=0.2,
          epochs=100,
          batch_size=256,
          verbose=1,
          callbacks=[early_stopping],
          class_weight=class_weight_dict
         )


print("Creating submission....")


# Use our trained model to make a prediction on the left-out test set
prediction_subm = model.predict(test).flatten()

# Create a submission file which will be exported to the output folder
submission = pd.DataFrame({'id': test.index, 'y': prediction_subm})
submission.to_csv('submission.csv', index=False)

