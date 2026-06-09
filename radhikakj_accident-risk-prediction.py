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


dataset_path='/kaggle/input/playground-series-s5e10'


import numpy as np
import pandas as pd
train_df=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train_df


train_df.info()


test_df=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_df.info()


train_df.columns


train_df['road_type'].value_counts()


train_df['lighting'].value_counts()


train_df['weather'].value_counts()


# Convert object columns → categorical
for col in train_df.select_dtypes('object').columns:
    train_df[col] = train_df[col].astype('category')



train_df['time_of_day'].value_counts()


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

cat_cols = train_df.select_dtypes('category').columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(train_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
train_df = pd.concat([train_df.drop(columns=cat_cols), encoded], axis=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = train_df.select_dtypes(['int64','float64']).columns
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])



from sklearn.feature_selection import mutual_info_regression
import numpy as np

X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']

mi = mutual_info_regression(X, y)
important_features = pd.Series(mi, index=X.columns).sort_values(ascending=False)



important_features


# Select top 15 important features
top_features = important_features.head(15).index

X_selected = X[top_features]
X_selected


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)



import tensorflow as tf
from tensorflow.keras import layers, models

# Build model
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # regression output
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    verbose=1
)



# Evaluate on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")



y_pred = model.predict(X_test)



import matplotlib.pyplot as plt

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Model Performance')
plt.show()



test_df.info()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models

# ------------------------------------------------------------
# ✅ STEP 1: Preprocess test_df — same as training
# ------------------------------------------------------------

# (1) Convert categorical columns to category
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cat_cols:
    test_df[col] = test_df[col].astype('category')

# (2) One-hot encode same columns as training
# ⚠️ Important: use same encoder as training; if not saved, re-fit on combined data
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit encoder on *train + test* categories if no saved encoder
encoder.fit(test_df[cat_cols])

# Encode test categorical features
encoded_test = pd.DataFrame(
    encoder.transform(test_df[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=test_df.index
)

# Drop original object columns & join encoded
test_processed = pd.concat([test_df.drop(columns=cat_cols), encoded_test], axis=1)

# (3) Scale numerical data
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
scaler = StandardScaler()
scaler.fit(test_df[num_cols])  # fit on training numerical data

test_processed[num_cols] = scaler.transform(test_processed[num_cols])

# ------------------------------------------------------------
# ✅ STEP 2: Select top features from training (mutual info)
# ------------------------------------------------------------
top_features = important_features.head(15).index  # or however many you chose

# Some features may not exist in test_df due to encoding — fix that:
for col in top_features:
    if col not in test_processed.columns:
        test_processed[col] = 0

# Align order
test_final = test_processed[top_features]

# ------------------------------------------------------------
# ------------------------------------------------------------
# ✅ STEP 3: Predict accident_risk
# ------------------------------------------------------------
y_pred_test = model.predict(test_final)
accident_risk_predicted = y_pred_test.flatten()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Reshape before scaling
accident_risk_normalized = scaler.fit_transform(accident_risk_predicted.reshape(-1, 1)).flatten()

# ------------------------------------------------------------
# ✅ STEP 4: Create submission DataFrame
# ------------------------------------------------------------
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': accident_risk_normalized
})




submission


# ------------------------------------------------------------
# ✅ STEP 5: Save to CSV
# ------------------------------------------------------------
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created successfully!")





