import numpy as np 
import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



train.head()


test.head()


# ===== CONVERT STRING CLASSES TO NUMERIC =====
# Convert categorical variables to numeric
train['Stage_fear'] = train['Stage_fear'].map({'No': 0, 'Yes': 1})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Handle missing values in categorical columns
train['Stage_fear'] = train['Stage_fear'].fillna(0)  # -1 for missing
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(0)  # -1 for missing

# Fill missing values in numerical columns with median
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                 'Friends_circle_size', 'Post_frequency']

for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def create_features(df):
    df_new = df.copy()
    
     # Convert categorical to numeric
    df_new['Stage_fear'] = df_new['Stage_fear'].map({'No': 0, 'Yes': 1}).fillna(-1)
    df_new['Drained_after_socializing'] = df_new['Drained_after_socializing'].map({'No': 0, 'Yes': 1}).fillna(-1)
    
    # Fill missing values
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                     'Friends_circle_size', 'Post_frequency']
    
    for col in numerical_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(df_new[col].median())
    
    # Advanced feature engineering with safe division
    df_new['social_energy_ratio'] = df_new['Social_event_attendance'] / (df_new['Time_spent_Alone'] + 1)
    df_new['friend_social_ratio'] = df_new['Friends_circle_size'] / (df_new['Social_event_attendance'] + 1)
    df_new['post_social_ratio'] = df_new['Post_frequency'] / (df_new['Social_event_attendance'] + 1)
    
    # Composite scores
    df_new['social_activity_score'] = (df_new['Social_event_attendance'] + df_new['Going_outside'] + 
                                      df_new['Friends_circle_size'] + df_new['Post_frequency']) / 4
    
    df_new['introvert_score'] = (df_new['Time_spent_Alone'] + 
                                df_new['Stage_fear'] + 
                                df_new['Drained_after_socializing']) / 3
    
    # Binary features
    df_new['high_social'] = (df_new['Social_event_attendance'] > 6).astype(int)
    df_new['low_friends'] = (df_new['Friends_circle_size'] < 5).astype(int)
    df_new['high_posts'] = (df_new['Post_frequency'] > 5).astype(int)
    df_new['high_alone_time'] = (df_new['Time_spent_Alone'] > 5).astype(int)
    
    # Polynomial features
    df_new['social_squared'] = df_new['Social_event_attendance'] ** 2
    df_new['friends_squared'] = df_new['Friends_circle_size'] ** 2
    df_new['alone_squared'] = df_new['Time_spent_Alone'] ** 2
    
    # Interaction features
    df_new['social_friend_product'] = df_new['Social_event_attendance'] * df_new['Friends_circle_size']
    df_new['alone_post_ratio'] = df_new['Time_spent_Alone'] / (df_new['Post_frequency'] + 1)
    df_new['going_outside_ratio'] = df_new['Going_outside'] / (df_new['Time_spent_Alone'] + 1)
    
    # Advanced ratios with safe division
    df_new['social_efficiency'] = df_new['Friends_circle_size'] / (df_new['Time_spent_Alone'] + 1)
    df_new['activity_balance'] = df_new['Social_event_attendance'] / (df_new['Post_frequency'] + 1)
    
    # Fix the problematic energy_management feature
    df_new['energy_management'] = np.where(
        df_new['Drained_after_socializing'] == -1,
        0,  # Default value for missing
        df_new['Going_outside'] / (df_new['Drained_after_socializing'] + 1)
    )
    
    # Categorical encodings
    df_new['stage_fear_encoded'] = df_new['Stage_fear'].map({-1: 0, 0: 1, 1: 2})
    df_new['drained_encoded'] = df_new['Drained_after_socializing'].map({-1: 0, 0: 1, 1: 2})
    
    # Handle any remaining infinite values
    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    
    # Fill any NaN values that might have been created
    for col in df_new.columns:
        if col not in ['id', 'Personality']:
            if df_new[col].dtype in ['float64', 'float32']:
                df_new[col] = df_new[col].fillna(df_new[col].median())
            else:
                df_new[col] = df_new[col].fillna(0)
    
    return df_new



# 2. Prepare data
print("Creating enhanced features...")
train_enhanced = create_features(train)
feature_cols = [col for col in train_enhanced.columns if col not in ['id', 'Personality']]

X = train_enhanced[feature_cols]
y = train_enhanced['Personality']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")



# 3. Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)



from tensorflow.keras import layers

def create_custom_ann(input_dim):
    model = keras.Sequential([
        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(input_dim,)),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.ELU(alpha=1.0),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='swish', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])
    return model


# 6. Compile and train model
print("Creating and training Custom ANN...")
model = create_custom_ann(X_train.shape[1])



# Compile with advanced optimizer and loss
optimizer = keras.optimizers.Adam(learning_rate=0.0001,clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)



lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_scheduler],
    verbose=1
)



# 7. Evaluate model
print("\n=== MODEL EVALUATION ===")
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

ann_accuracy = accuracy_score(y_test, y_pred)
print(f"Custom ANN Accuracy: {ann_accuracy:.4f}")



# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# 8. Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# 9. Feature importance using permutation importance
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

class ANNWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()

def get_feature_importance(model, X, y, feature_names):
    # Create a wrapper for the ANN
    ann_wrapper = ANNWrapper(model)
    
    # Calculate permutation importance
    result = permutation_importance(
        estimator=ann_wrapper,
        X=X,
        y=y,
        scoring='accuracy',
        n_repeats=10,
        random_state=42
    )
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    }).sort_values('Importance', ascending=False)
    
    return importance_df


# Get feature importance
print("\nCalculating feature importance...")
feature_importance = get_feature_importance(model, X_test, y_test, feature_cols)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))



# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                color=['#FF6B6B' if x < 0.1 else '#4ECDC4' for x in top_features['Importance']])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance for Custom ANN', fontsize=16, fontweight='bold')

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.show()


# 10. Cross-validation
print("\nPerforming cross-validation...")
from sklearn.model_selection import KFold

def cross_validate_ann(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model for this fold
        fold_model = create_custom_ann(X_fold_train.shape[1])
        fold_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with early stopping
        fold_model.fit(
            X_fold_train, y_fold_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )
        
        # Evaluate
        y_fold_pred = (fold_model.predict(X_fold_val) > 0.5).astype(int).flatten()
        fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
        scores.append(fold_accuracy)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    return scores


cv_scores = cross_validate_ann(X_scaled, y)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")



# 11. Predict on test set
print("\nPredicting on test set...")
test_enhanced = create_features(test)
X_test_final = test_enhanced[feature_cols]
X_test_final_scaled = scaler.transform(X_test_final)

test_predictions_proba = model.predict(X_test_final_scaled)
test_predictions = (test_predictions_proba > 0.5).astype(int).flatten()



# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_predictions
})
submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")



# 12. Final summary
print("\n" + "="*50)
print("CUSTOM ANN MODEL SUMMARY")
print("="*50)
print(f"Test Accuracy: {ann_accuracy:.4f}")
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"Regularization: L2 + Dropout + BatchNorm")
print(f"Optimizer: Adam with learning rate scheduling")
print("="*50) 

