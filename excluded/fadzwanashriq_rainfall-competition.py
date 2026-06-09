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


import pandas as pd
import numpy as np

#data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
import seaborn as sns


train_path = '/kaggle/input/playground-series-s5e3/train.csv'
test_path = '/kaggle/input/playground-series-s5e3/train.csv'
submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.head()


train_df.info()


train_df.corr()




# Check for missing values
print(train_df.isnull().sum())

# Summary statistics
print(train_df.describe())

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# Regression


train_df.drop(columns=['id'], inplace = True)
test_df.drop(columns=['id'], inplace = True)

print(train_df.isnull().sum())

X = train_df.drop(columns=['temparature'])
y = train_df['temparature']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)

r2 = r2_score(y_val, y_pred)

print(f'Mean Squared Error:{mse}')
print(f'R-squared: {r2}')


from sklearn.ensemble import RandomForestRegressor 

#initialize and train a random forest model
rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(X_train, y_train)

#evaluate the random forest model
rf_y_pred = rf_model.predict(X_val)
rf_mse = mean_squared_error(y_val, rf_y_pred)
rf_r2 = r2_score(y_val, rf_y_pred)

print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest R-squared: {rf_r2}')


from tensorflow import keras
import tensorflow as tf 
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

class Config:
    train_link = train_path
    test_link = test_path
    sub_link = submission_path
    original = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"
    


train = pd.read_csv(Config.train_link, index_col='id').fillna(-1)
test = pd.read_csv(Config.test_link, index_col='id').fillna(-1)  # Ensure no NaN issues

combined = pd.concat([train, test], axis=0, ignore_index=True)

# Check if both 'windspeed' and 'winddirection' exist BEFORE popping
if {'windspeed', 'winddirection'}.issubset(combined.columns):
    windspeed = combined.pop('windspeed')
    wd_rad = np.radians(combined.pop('winddirection'))  # Convert degrees to radians

    # Calculate wind vector components
    combined['Wx'] = windspeed * np.cos(wd_rad)
    combined['Wy'] = windspeed * np.sin(wd_rad)
else:
    raise KeyError("Missing 'windspeed' or 'winddirection' column in dataset")



# Split processed data back into train and test sets
train = combined.iloc[:len(train)]
test = combined.iloc[len(train):].drop(columns=['rainfall'], errors='ignore')

# Define features and target variable
X = train.drop(columns=['rainfall'])
y = train['rainfall']

print(f"Train Shape: {X.shape}, Test Shape: {test.shape}, Target Shape: {y.shape}")



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=42)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mean = x_train.mean(axis =0)
std = x_train.std(axis =0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to NumPy and reshape
num_features = x_train.shape[1]  # Dynamically get feature count
x_train = x_train.reshape((x_train.shape[0], 1, num_features))
x_test = x_test.reshape((x_test.shape[0], 1, num_features))

print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    MultiHeadAttention,
    Dropout,
    LayerNormalization,
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Input,
    Add,
)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """Transformer Encoder Block with Multi-Head Attention, Normalization, and Feedforward Layers."""
    
    # Multi-Head Self Attention + Residual Connection
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    x = Add()([inputs, attention_output])  # Residual connection

    # Feedforward Network + Residual Connection
    ff_output = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
    return Add()([x, ff_output])  # Residual connection




def build_model(
    input_shape,
    head_size=64,
    num_heads=2,
    ff_dim=128,
    num_transformer_blocks=2,
    mlp_units=[64],
    dropout=0.0,
    mlp_dropout=0.0,
):
    """Builds a Transformer-based model for sequence data processing."""
    
    inputs = Input(shape=input_shape)
    x = inputs

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global Pooling & MLP Head
    x = GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(1, activation="sigmoid")(x)  # Binary classification output
    return keras.Model(inputs, outputs)




# Define input shape
input_shape = x_train.shape[1:]

# Build Transformer model
model = build_model(
    input_shape=input_shape,
    head_size=256,
    num_heads=12,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25
)

# Compile model with Adam optimizer and AUC metric
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[tf.keras.metrics.AUC(name="AUC")],  # Correct AUC metric usage
)

# Print model summary
model.summary()

# Define callbacks for better training stability
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )
]

# Train the model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,  # Make sure dataset is large enough for this
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,  # More readable logs
)

# Evaluate the model on test data
test_loss, test_auc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")



import pandas as pd
import numpy as np

# # Normalize test data (avoid modifying the original test DataFrame)
test_scaled = (test - test.mean(axis=0)) / test.std(axis=0)
test_scaled = test_scaled.to_numpy().astype(np.float32)  # Convert to NumPy array for efficiency

# # Reshape test data for model input
test_scaled = test_scaled.reshape(test_scaled.shape[0], 1, 11)

# Make predictions
preds = model.predict(test_scaled).flatten()  # Flatten to 1D array for submission
print(preds)

# Prepare submission file
submission = pd.DataFrame({"id": test.index, "rainfall": preds})
submission = submission.iloc[:730] 
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' saved successfully!")



import os

# Automatically submit the file using Kaggle API
os.system("kaggle competitions submit -c playground-series-s5e3 -f submission.csv -m 'Auto submission'")

print("Submission successful!")



!kaggle competitions submit -c playground-series-s5e3 -f submission.csv -m "Auto submission"

