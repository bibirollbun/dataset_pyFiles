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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


print("Train Shape:",train.shape)
print("Test Shape:",test.shape)


train.isnull().sum()


test.isnull().sum()


train.columns


test.columns


## dropping id feature from train and test 
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


train.columns


train.info()


train.describe()


## importing required libraries
import seaborn as sns
import matplotlib.pyplot as plt 


def filter_outliers(df,feature):
    ## using z score method
    std = df[feature].std()
    average = df[feature].mean()

    ## lower bound
    lower_bound = average - 3*std
    upper_bound = average + 3*std

    df = df[(df[feature]>=lower_bound) & (df[feature]<=upper_bound)]
    return df


def return_outliers_using_zscore(df,feature):
    ## using z score method
    std = df[feature].std()
    average = df[feature].mean()

    ## lower bound
    lower_bound = average - 3*std
    upper_bound = average + 3*std

    outliers_in_df = df[(df[feature]<lower_bound) | (df[feature]>upper_bound)]
    return outliers_in_df


for i,feature in enumerate(train.columns,1):
    if train[feature].dtype!='O':
        print(f"\n{feature}--->")
        display(return_outliers_using_zscore(train,feature))
        print()
        if(i==1):
            outliers_df = return_outliers_using_zscore(train,feature)
        outliers_df = pd.concat([outliers_df,return_outliers_using_zscore(train,feature)])
        # outliers_df = pd.concat([outliers_df,return_outliers_using_zscore(train,feature)],ignore_index=True)
    else:
        print(f"{feature} is a categorical feature.....\n")


outliers_df.shape


outliers_df[outliers_df.duplicated()]


outliers_df = outliers_df.drop_duplicates()


outliers_df.shape


outliers_df[outliers_df.duplicated()]


outliers_df


train = train.drop(outliers_df.index)


train


# for feature in train.columns:
#     if train[feature].dtype!='O':
#         print(f"\n{feature}--->")
#         display(return_outliers_using_zscore(train,feature))
#         print()
#     else:
#         print(f"{feature} is a categorical feature.....\n")
        


# def remove_outliers(df):
#     for feature in df.columns:
#         if df[feature].dtype!='O': ## if feature is not object types means numeric then
#             df = filter_outliers(df,feature)  
#     return df


train.shape


train


train[train['balance']<0] ## rows where balance is in negative


# def preprocess_data(df):
#     # Create a new feature to handle the -1 in pdays, which indicates no previous contact
#     df['pdays_was_contacted'] = (df['pdays'] != -1).astype(int)

#     # Transform highly skewed numerical features with log1p
#     # Log transformation helps normalize the distribution, which can improve model performance
#     for col in ['campaign', 'pdays', 'previous']:
#         if col in df.columns:
#             df[col] = np.log1p(df[col]) + 1 # Adding 1 to avoid issues with zero/negative values

#     return df


# --------------------------
def preprocess_data(df):
    # Create a new feature to handle the -1 in pdays
    df['pdays_was_contacted'] = (df['pdays'] != -1).astype(int)
    
    # Handle negative 'balance' values before log transformation
    # A common and effective strategy is to create a new binary feature
    # to flag if the balance was negative, then transform the absolute value.
    if 'balance' in df.columns:
        df['balance_is_negative'] = (df['balance'] < 0).astype(int)
        df['balance'] = df['balance'].abs() # Take the absolute value for log transformation
        df['balance'] = np.log1p(df['balance'])
        
    # Transform other highly skewed numerical features with log1p
    for col in ['campaign', 'pdays', 'previous']:
        if col in df.columns:
            # We add 1 to the column value before taking log1p to ensure the transformation is valid.
            # This is a robust way to handle features that may contain 0s.
            df[col] = np.log1p(df[col] + 1)
            
    return df


train = preprocess_data(train)
test = preprocess_data(test)


train.head()


# Identify categorical and numerical features
num_cols = list(train.select_dtypes(exclude=['object']).columns.difference(['y']))
cat_cols = list(train.select_dtypes(include=['object']).columns)


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


for feature in cat_cols:
    train[feature] = le.fit_transform(train[feature])
    test[feature] = le.transform(test[feature])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# for num_feature in num_cols:
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True)
plt.show()


X = train.drop('y',axis=1)
y = train['y']


X.shape


X


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Model definition
# ==============================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Output layer

# ==============================
# Compile model
# ==============================
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==============================
# Add EarlyStopping callback
# ==============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,          # Stop if no improvement for 5 epochs
    restore_best_weights=True # Roll back to best weights
)

# ==============================
# Training with EarlyStopping
# ==============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    verbose=1,
    callbacks=[early_stop]
)

# ==============================
# Evaluate on test set
# ==============================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# ==============================
# Predictions & ROC-AUC
# ==============================
y_prob = model.predict(X_test)   # probabilities for ROC-AUC
y_pred = (y_prob > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ==============================
# Plot Accuracy & Loss
# ==============================
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.show()

# ==============================
# Plot ROC Curve
# ==============================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


## prediction on test data
test_pred = model.predict(test) ## for class 1 probabilities 

## saving prediction in submission file
sample_submission['y'] = test_pred
sample_submission.to_csv(f"ann_prediction.csv",index=False)
display(sample_submission.head())
print(f"File saved as ann_prediction.csv.....\n")



test_pred







