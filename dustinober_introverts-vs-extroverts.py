import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# --- 1. Load and Preprocess Data ---
# For reproducibility
np.random.seed(12)
tf.random.set_seed(12)

# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Prepare data
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
shared_cols = list(set(X.columns) & set(test_df.columns))
X = X[shared_cols]
test_df = test_df[shared_cols]

# Impute and Encode
for col in X.columns:
    if X[col].dtype == 'object':
        X[col].fillna(X[col].mode()[0], inplace=True)
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)
        test_df[col].fillna(test_df[col].median(), inplace=True)

le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])


# --- 2. Train Model 1: Random Forest ---
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=2, random_state=12, n_jobs=-1)
rf_model.fit(X, y_encoded)
rf_probs = rf_model.predict_proba(test_df)[:, 1]
print("Random Forest training complete.")


# --- 3. Train Model 2: Neural Network ---
print("\nTraining Neural Network model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_scaled.shape[1]]),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy')
nn_model.fit(X_scaled, y_encoded, epochs=40, batch_size=32, verbose=0)
nn_probs = nn_model.predict(test_scaled).flatten()
print("Neural Network training complete.")


# --- 4. Train Model 3: LightGBM ---
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMClassifier(random_state=12)
lgb_model.fit(X, y_encoded)
lgb_probs = lgb_model.predict_proba(test_df)[:, 1]
print("LightGBM training complete.")


# --- 5. Blend Predictions and Create Submission ---
print("\nBlending predictions from all three models...")
# Average the probabilities from all three models
blended_probs = (rf_probs + nn_probs + lgb_probs) / 3.0

# Convert blended probabilities to final 0/1 predictions
blended_predictions = (blended_probs > 0.5).astype(int)

# Inverse transform to get 'Extrovert'/'Introvert'
final_predictions = le_y.inverse_transform(blended_predictions)

# Create the submission file
submission_df_final_blend = pd.DataFrame({'id': test_df['id'], 'Personality': final_predictions})
submission_df_final_blend.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Final blended submission file head:")
print(submission_df_final_blend.head())




