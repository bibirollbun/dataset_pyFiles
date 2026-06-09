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


import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df= df.drop(['id', 'day'], axis= 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


#WINSORIZATION

from scipy.stats.mstats import winsorize

# List of numerical columns to apply Winsorization
outlier_cols = ['dewpoint', 'humidity', 'cloud']

# Apply Winsorization (Capping at 1st and 99th percentile)
for col in outlier_cols:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])  # 1% from both tails

# Check if outliers are reduced
df[outlier_cols].describe()


from scipy.stats import boxcox

# Define transformations for each type of skewness
right_skewed_cols = ['pressure', 'sunshine', 'winddirection', 'windspeed']
left_skewed_cols = ['maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud']

# Apply log transformation for right-skewed columns (add 1 to avoid log(0) errors)
for col in right_skewed_cols:
    df[col] = np.log1p(df[col])  # log1p is log(1 + x), safer for small values

# Apply power transformation (square) for left-skewed columns
for col in left_skewed_cols:
    df[col] = np.power(df[col], 2)  # Square the values to normalize left skew

df['cloud'] = np.power(df['cloud'], 2)
# Check skewness again
df.skew()


# some new features
df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)
df['relative_dryness'] = 100 - df['humidity']
df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])


df.columns


df.skew()


# Calculate correlation matrix
corr_matrix = df[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'winddirection', 'windspeed',
       'humidity_cloud_interaction',
       'humidity_sunshine_interaction', 'cloud_sunshine_ratio',
       'relative_dryness', 'sunshine_percentage', 'weather_index', 'rainfall']].corr()

# Create heatmap
plt.figure(figsize=(10, 8))  # Adjust size as needed
sns.heatmap(corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center the colormap at 0
            square=True,  # Make the plot square-shaped
            linewidths=.5,  # Width of the lines between cells
            fmt='.2f')  # Format annotation with 2 decimal places

plt.title('Correlation Matrix Heatmap')
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# ===========================
# Load and preprocess dataset
# ===========================

X = df.drop(columns=['rainfall'])  
y = df['rainfall']

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale the features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================


# Build ANN Architecture
# ======================

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# Improved ANN model architecture
model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(16,), 
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.4),
    
    # Hidden layers with increasing complexity
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output layer
    Dense(1, activation='sigmoid')
])

# Compile with a slightly lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name="AUC", curve='ROC'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Advanced callbacks
early_stopping = EarlyStopping(
    monitor='val_AUC', 
    patience=25,
    restore_best_weights=True,
    verbose=1,
    mode='max'  # Using max because we want to maximize AUC
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_AUC',
    factor=0.2,
    patience=10,
    min_lr=1e-6,
    verbose=1,
    mode='max'  # Using max because we want to maximize AUC
)

# Training with class weights if data is imbalanced
# Calculate class weights if your dataset is imbalanced
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Uncomment and use this if your classes are imbalanced
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train the model (with or without class weights)
history = model.fit(
    X_train, y_train,
    epochs=150,  # Increase max epochs, early stopping will prevent overfitting
    batch_size=16,  # Smaller batch size for better generalization
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    # class_weight=class_weight_dict,  # Uncomment if using class weights
    verbose=1
)


# Predict probabilities for both classes
y_pred_proba = model.predict(X_test)  # Probabilities (NOT class labels)

# Compute AUC-ROC using probabilities
auc_score = roc_auc_score(y_test, y_pred_proba)

# Convert probabilities to class labels (Threshold = 0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print evaluation metrics
print(f"AUC Score: {auc_score:.4f}")


# ==================
# Confusion Matrix
# ==================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==================
# ROC Curve Plot
# ==================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



df_test= pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test= df_test.drop(['id', 'day'], axis=1)
df_test["winddirection"].fillna(df_test["winddirection"].mean(), inplace=True)

from scipy.stats import boxcox

# Define transformations for each type of skewness
right_skewed_cols = ['pressure', 'sunshine', 'winddirection', 'windspeed']
left_skewed_cols = ['maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud']

# Apply log transformation for right-skewed columns (add 1 to avoid log(0) errors)
for col in right_skewed_cols:
    df_test[col] = np.log1p(df_test[col])  # log1p is log(1 + x), safer for small values

# Apply power transformation (square) for left-skewed columns
for col in left_skewed_cols:
    df_test[col] = np.power(df_test[col], 2)  # Square the values to normalize left skew

df_test['cloud'] = np.power(df_test['cloud'], 2)  # Square the values to normalize left skew

# Check skewness again
df_test.skew()

# some new features
df_test['humidity_cloud_interaction'] = df_test['humidity'] * df_test['cloud']
df_test['humidity_sunshine_interaction'] = df_test['humidity'] * df_test['sunshine']
df_test['cloud_sunshine_ratio'] = df_test['cloud'] / (df_test['sunshine'] + 1e-5)
df_test['relative_dryness'] = 100 - df_test['humidity']
df_test['sunshine_percentage'] = df_test['sunshine'] / (df_test['sunshine'] + df_test['cloud'] + 1e-5)
df_test['weather_index'] = (0.4 * df['humidity']) + (0.3 * df_test['cloud']) - (0.3 * df_test['sunshine'])

# Standardize features
scaler = StandardScaler()
df_test = scaler.fit_transform(df_test)


pred_prob= model.predict(df_test)

test_data_id = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')  # Replace with your test file path
submission_df = pd.DataFrame({
    'id': test_data_id['id'],    # Extract 'id' column from df_test
    'rainfall': pred_prob.ravel()       # Assign predictions to 'rainfall' column
})

submission_df.to_csv('submission.csv', index= False)

