# Step 1: Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load


np.random.seed(42)
tf.random.set_seed(42)


df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
data_description = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


def analyze_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage": missing_pct,
    })
    return missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)

# Example usage:
display(analyze_missing_values(df))


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='efs')
plt.title('Distribution of Event-Free Survival')
plt.show()


def engineer_features(df):
    """
    Feature engineering function for the given dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: Transformed DataFrame with new features.
    """
    df = df.copy()
    
    # Example age-like feature (replace if age_at_hct exists in your data)
    if 'efs_time' in df.columns:
        df['age_group'] = pd.qcut(df['efs_time'], q=5, 
                                  labels=['VeryYoung', 'Young', 'Middle', 'Old', 'VeryOld'])
    
    # HLA matching score
    hla_cols = [col for col in df.columns if 'hla' in col and 'match' in col]
    if hla_cols:
        df['hla_match_score'] = df[hla_cols].mean(axis=1)
    
    # Risk categorization based on a relevant score
    if 'efs_time' in df.columns:
        df['risk_group'] = df['efs_time'].apply(
            lambda x: 'Low' if x <= 20 else ('Medium' if x <= 50 else 'High'))
    
    # Interaction features
    if 'efs_time' in df.columns and 'hla_match_c_high' in df.columns:
        df['efs_hla_interaction'] = df['efs_time'] * df['hla_match_c_high']
    
    return df

# Apply feature engineering
# Ensure train_df and test_df are defined before running this script
train_df = engineer_features(df)
test_df = engineer_features(df)

print("New features created. Updated shape:", df.shape)


def preprocess_data(train_df, test_df):
    # Separate target
    y = train_df['efs']
    X = train_df.drop(['efs', 'efs_time'], axis=1)
    X_test = test_df.copy()

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Separate numerical and categorical columns
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Transform data
    X_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test)

    return X_processed, y, X_test_processed, preprocessor

# Apply preprocessing
X_processed, y, X_test_processed, preprocessor = preprocess_data(df, test)
print("Processed data shapes:", X_processed.shape, X_test_processed.shape)


def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='selu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='selu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='selu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model

# Train with K-fold cross-validation
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
models = []
histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_processed, y)):
    print(f'Training fold {fold + 1}/{n_splits}')

    X_train_fold = X_processed[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_processed[val_idx]
    y_val_fold = y[val_idx]

    model = build_model(X_train_fold.shape[1])

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6),
        callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    ]

    # Train
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks_list,
        verbose=1
    )

    models.append(model)
    histories.append(history)



# Plot training history
def plot_training_history(histories):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Train (Fold {i+1})')
        plt.plot(history.history['val_loss'], label=f'Val (Fold {i+1})')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Train (Fold {i+1})')
        plt.plot(history.history['val_accuracy'], label=f'Val (Fold {i+1})')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(histories)

