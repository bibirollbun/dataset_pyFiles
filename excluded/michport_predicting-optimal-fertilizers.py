import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Data Loading
data_dir = '/kaggle/input/playground-series-s5e6/'
try:
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sample_submission_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    print("Data loaded successfully from directory.")
except FileNotFoundError:
    print("Error: Data files not found. Please ensure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the directory.")
    exit()

print("\n--- Train DataFrame Head: ---")
print(train_df.head())
print("\n--- Test DataFrame Head: ---")
print(test_df.head())

# Summaries
print("\n--- Train DataFrame Descriptive Statistics: ---")
print(train_df.describe())
print("\n--- Test DataFrame Descriptive Statistics: ---")
print(test_df.describe())

# Missing Values
print("\n--- Missing values in Train DataFrame: ---")
print(train_df.isnull().sum())
print("\n--- Missing values in Test DataFrame: ---")
print(test_df.isnull().sum())



numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'ID' if present and target 'Fertilizer Name'
if 'ID' in numerical_features:
    numerical_features.remove('ID')
if 'id' in numerical_features:  # Corrected column name based on output
    numerical_features.remove('id')


print("\n--- Distributions of Numerical Features: ---")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 4, i + 1)
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()


categorical_features = train_df.select_dtypes(include='object').columns.tolist()
if 'Fertilizer Name' in categorical_features:
    categorical_features.remove('Fertilizer Name') # Target variable

print("\n--- Distributions of Categorical Features (if any - Target variable distribution shown below): ---")

# Target Variable Distribution
plt.figure(figsize=(8, 6))
sns.countplot(y=train_df['Fertilizer Name'], order=train_df['Fertilizer Name'].value_counts().index)
plt.title('Distribution of Target Variable: Fertilizer Name')
plt.show()
print("\nTarget variable counts:")
print(train_df['Fertilizer Name'].value_counts())


print("\n--- Correlation Matrix of Numerical Features: ---")
correlation_matrix = train_df[numerical_features].corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Separate features and target
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']
X_test = test_df.drop('id', axis=1)

# Identify numerical and categorical features
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Number of target classes: {num_classes}")

# MAP@3 Evaluation Metric (Mean Average Precision at 3)
# Custom metric function as required by the competition
def map_at_3(y_true, y_pred_proba):
    """
    Calculates Mean Average Precision at 3 (MAP@3).
    y_true: True labels (encoded integers).
    y_pred_proba: Predicted probabilities for each class.
    """
    # Get the top 3 predicted class indices for each sample
    # sort in desc order of probability and take top 3 idxs
    top_3_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]

    num_samples = y_true.shape[0]
    avg_precision_sum = 0.0

    for i in range(num_samples):
        true_label = y_true[i]
        predicted_labels_at_k = top_3_indices[i]

        precision_at_k = 0.0
        num_hits = 0
        for k_idx, predicted_label in enumerate(predicted_labels_at_k):
            if predicted_label == true_label:
                num_hits += 1
                # Precision at k: (Number of hits up to k) / k+1
                precision_at_k += num_hits / (k_idx + 1)
                break # Found the true label, no need to check further in top 3
        if num_hits > 0: # Only add if the true label was found in the top 3
            avg_precision_sum += precision_at_k

    return avg_precision_sum / num_samples if num_samples > 0 else 0.0


def create_mlp_model_1(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # Accuracy for monitoring during training
    return model


def create_mlp_model_2(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.00001, verbose=1)

models_results = {}
all_test_preds_proba_model1 = []
all_test_preds_proba_model2 = []


print(f"\nTraining Models with {N_SPLITS}-Fold Stratified Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

    # Apply preprocessing pipeline
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_val_processed = preprocessor.transform(X_val_fold)
    # Important: Transform test data using the preprocessor fitted on the *training data of the current fold*
    X_test_processed = preprocessor.transform(X_test) 

    input_shape = X_train_processed.shape[1]

    print("Training Model 1 (Simple MLP)...")
    model1 = create_mlp_model_1(input_shape, num_classes)
    history1 = model1.fit(X_train_processed, y_train_fold,
                          epochs=100,
                          batch_size=32,
                          validation_data=(X_val_processed, y_val_fold),
                          callbacks=[early_stopping, reduce_lr],
                          verbose=0)

    val_loss_model1, val_acc_model1 = model1.evaluate(X_val_processed, y_val_fold, verbose=0)
    print(f"Model 1 - Fold {fold+1} Validation Accuracy: {val_acc_model1:.4f}")

    # Predict probabilities for validation set for MAP@3 calculation
    y_val_pred_proba_model1 = model1.predict(X_val_processed)
    fold_map3_model1 = map_at_3(y_val_fold, y_val_pred_proba_model1)
    print(f"Model 1 - Fold {fold+1} MAP@3: {fold_map3_model1:.4f}")

    # Predict probabilities for test set for ensemble/average prediction later
    test_preds_proba_model1 = model1.predict(X_test_processed)
    all_test_preds_proba_model1.append(test_preds_proba_model1)

    # --- Model 2 Training ---
    print("Training Model 2 (Deeper MLP)...")
    model2 = create_mlp_model_2(input_shape, num_classes)
    history2 = model2.fit(X_train_processed, y_train_fold,
                          epochs=100,
                          batch_size=32,
                          validation_data=(X_val_processed, y_val_fold),
                          callbacks=[early_stopping, reduce_lr],
                          verbose=0)

    val_loss_model2, val_acc_model2 = model2.evaluate(X_val_processed, y_val_fold, verbose=0)
    print(f"Model 2 - Fold {fold+1} Validation Accuracy: {val_acc_model2:.4f}")

    y_val_pred_proba_model2 = model2.predict(X_val_processed)
    fold_map3_model2 = map_at_3(y_val_fold, y_val_pred_proba_model2)
    print(f"Model 2 - Fold {fold+1} MAP@3: {fold_map3_model2:.4f}")

    test_preds_proba_model2 = model2.predict(X_test_processed)
    all_test_preds_proba_model2.append(test_preds_proba_model2)

    if 'Model 1 MAP@3' not in models_results:
        models_results['Model 1 MAP@3'] = []
        models_results['Model 2 MAP@3'] = []
    models_results['Model 1 MAP@3'].append(fold_map3_model1)
    models_results['Model 2 MAP@3'].append(fold_map3_model2)


print("\n--- Results and Discussion ---")

# Calculate average MAP@3
avg_map3_model1 = np.mean(models_results['Model 1 MAP@3'])
avg_map3_model2 = np.mean(models_results['Model 2 MAP@3'])

print(f"\nAverage MAP@3 for Model 1 (Simple MLP) across {N_SPLITS} folds: {avg_map3_model1:.4f}")
print(f"Average MAP@3 for Model 2 (Deeper MLP) across {N_SPLITS} folds: {avg_map3_model2:.4f}")

# Model Comparison
print("\nModel Comparison:")
print(f"Model 1 MAP@3 scores per fold: {[f'{score:.4f}' for score in models_results['Model 1 MAP@3']]}")
print(f"Model 2 MAP@3 scores per fold: {[f'{score:.4f}' for score in models_results['Model 2 MAP@3']]}")

if avg_map3_model1 > avg_map3_model2:
    print("\nModel 1 (Simple MLP) performed better on average.")
elif avg_map3_model2 > avg_map3_model1:
    print("\nModel 2 (Deeper MLP) performed better on average.")
else:
    print("\nBoth models performed similarly on average.")


# vAverage probabilities from all folds for both models
# This acts as a simple ensemble strategy
ensemble_test_preds_proba = (np.mean(all_test_preds_proba_model1, axis=0) +
                             np.mean(all_test_preds_proba_model2, axis=0)) / 2

# Get top 3 predicts for each sample
top_3_indices = np.argsort(ensemble_test_preds_proba, axis=1)[:, ::-1][:, :3]

# Convert idxs back to original fertilizer names
predicted_fertilizer_names = label_encoder.inverse_transform(top_3_indices.flatten()).reshape(-1, 3)

# Format for submission
submission_predictions = [' '.join(row) for row in predicted_fertilizer_names]

# Create the submission DataFrame with the correct column name
submission_df = pd.DataFrame({'ID': test_df['id'], 'Fertilizer Name': submission_predictions})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' generated successfully!")
print("\nSample of generated submission file:")
print(submission_df.head())

