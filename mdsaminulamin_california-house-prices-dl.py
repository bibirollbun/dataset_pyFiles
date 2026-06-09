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


train_df = pd.read_csv('/kaggle/input/california-house-prices/train.csv')
test_df = pd.read_csv('/kaggle/input/california-house-prices/test.csv')


train_df.shape


train_df.info()


train_df.select_dtypes(include=['object']).nunique()


train_df[['Listed On', 'Last Sold On']].head()


train_df.isna().mean().sort_values(ascending=False)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


!pip install tensorflow


# TensorFlow imports - use this specific pattern
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    print("TensorFlow imported successfully")
except Exception as e:
    print(f"TensorFlow import error: {e}")


# import tensorflow as tf


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.metrics import mean_squared_error


def create_features(df):
    df = df.copy()
    
    # Price per square foot
    df['price_per_sqft'] = df['Listed Price'] / df['Total interior livable area']
    
    # Age of property (using 2021 as competition year)
    df['property_age'] = 2021 - df['Year built']
    
    # School score average
    school_cols = ['Elementary School Score', 'Middle School Score', 'High School Score']
    df['avg_school_score'] = df[school_cols].mean(axis=1)
    
    return df

# Before preprocessing, add:
train_df = create_features(train_df)

# Before preprocessing test data, add:
test_df = create_features(test_df)


# Preprocessing strategy - Modified to fix the error
def preprocess_data(train_df, test_df=None, numeric_imputer=None, categorical_imputer=None, scaler=None, label_encoders=None, is_training=True):
    if is_training:
        # Separate features and target for training
        X = train_df.drop(['Sold Price', 'Id', 'Address'], axis=1)
        y = train_df['Sold Price']
    else:
        # For test data
        X = test_df.drop(['Id', 'Address'], axis=1)
        y = None
    
    # Handle high-cardinality categorical columns - reduce or drop
    high_cardinality_cols = ['Summary', 'Elementary School', 'Middle School', 'High School']
    X = X.drop(high_cardinality_cols, axis=1)
    
    # Separate numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    if is_training:
        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Scale numeric features
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        return X, y, numeric_imputer, categorical_imputer, scaler, label_encoders
    
    else:
        # For test data - use fitted imputers and encoders
        X[numeric_features] = numeric_imputer.transform(X[numeric_features])
        X[categorical_features] = categorical_imputer.transform(X[categorical_features])
        
        # Encode categorical variables using fitted label encoders
        for col in categorical_features:
            le = label_encoders[col]
            # Handle unseen categories in test set
            unique_vals = set(X[col].astype(str))
            unseen_vals = unique_vals - set(le.classes_)
            if unseen_vals:
                # For unseen categories, use the most frequent category
                X[col] = X[col].astype(str)
                X.loc[X[col].isin(unseen_vals), col] = le.classes_[0]
            X[col] = le.transform(X[col].astype(str))
        
        # Scale numeric features using fitted scaler
        X[numeric_features] = scaler.transform(X[numeric_features])
        
        return X, y


# Build optimal neural network
def create_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1, activation='linear')
    ])
    
    return model


# Callbacks
def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]


# Main training pipeline
def train_and_predict(train_df, test_df):
    # Preprocess training data
    print("Preprocessing training data...")
    X_train, y_train, numeric_imputer, categorical_imputer, scaler, label_encoders = preprocess_data(train_df, is_training=True)
    
    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_model(X_tr.shape[1])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Preprocess test data
    print("Preprocessing test data...")
    X_test, _ = preprocess_data(None, test_df, numeric_imputer, categorical_imputer, scaler, label_encoders, is_training=False)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Sold Price': predictions.flatten()
    })
    
    # Save to CSV
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")
    
    return model, history, submission


# Run the pipeline
model, history, submission = train_and_predict(train_df, test_df)




