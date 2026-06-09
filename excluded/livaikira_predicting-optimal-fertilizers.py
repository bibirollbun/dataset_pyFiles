%load_ext cudf.pandas

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as XGBoost
import lightgbm as lgb

# For reproducibility
import random
random.seed(42)
np.random.seed(42)


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# Display the first few rows of each dataset
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data preview:")
display(train_df.head())
print("\nTest data preview:")
display(test_df.head())


# Check for missing values
print("Missing values in training data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())

# Summary statistics
print("\nTraining data summary statistics:")
display(train_df.describe())

# Information about data types
print("\nTraining data types:")
print(train_df.dtypes)


# Analyze the target variable distribution
fertilizer_counts = train_df['Fertilizer Name'].value_counts()
print("Fertilizer distribution:")
display(fertilizer_counts)

# Plot the distribution of fertilizers
plt.figure(figsize=(12, 6))
sns.barplot(x=fertilizer_counts.index, y=fertilizer_counts.values)
plt.title('Distribution of Fertilizer Types')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Explore categorical features
plt.figure(figsize=(15, 10))

# Plot 1: Soil Type vs Fertilizer
plt.subplot(2, 2, 1)
soil_fert_counts = train_df.groupby(['Soil Type', 'Fertilizer Name']).size().unstack()
soil_fert_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Fertilizer Distribution by Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Crop Type vs Fertilizer
plt.subplot(2, 2, 2)
crop_fert_counts = train_df.groupby(['Crop Type', 'Fertilizer Name']).size().unstack()
crop_fert_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Fertilizer Distribution by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Explore numerical features
plt.figure(figsize=(20, 15))
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='Fertilizer Name', y=feature, data=train_df)
    plt.title(f'{feature} by Fertilizer Type')
    plt.xticks(rotation=90)
    
plt.tight_layout()
plt.show()


# Define features and target
X_train = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y_train = train_df['Fertilizer Name']
X_test = test_df.drop(['id'], axis=1)

# Process categorical features using one-hot encoding
categorical_features = ['Soil Type', 'Crop Type']
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create a copy for preprocessing
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Apply One-Hot Encoding to categorical features
X_combined = pd.concat([X_train[categorical_features], X_test[categorical_features]], axis=0)
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(X_combined)
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

encoded_df = pd.DataFrame.sparse.from_spmatrix(
    encoded_features,
    columns=encoded_feature_names,
    index=list(X_train.index) + list(X_test.index)
)

# Split back to train and test
X_train_encoded = encoded_df.iloc[:len(X_train)]
X_test_encoded = encoded_df.iloc[len(X_train):]

# Drop original categorical features and join encoded ones
X_train_processed = X_train.drop(categorical_features, axis=1)
X_test_processed = X_test.drop(categorical_features, axis=1)

X_train_processed = pd.concat([X_train_processed, X_train_encoded], axis=1)
X_test_processed = pd.concat([X_test_processed, X_test_encoded], axis=1)

print("Processed training features shape:", X_train_processed.shape)
print("Processed test features shape:", X_test_processed.shape)
display(X_train_processed.head())


# Scale numerical features
scaler = StandardScaler()
X_train_processed[numerical_features] = scaler.fit_transform(X_train_processed[numerical_features])
X_test_processed[numerical_features] = scaler.transform(X_test_processed[numerical_features])

# Encode the target variable (fertilizer names) into numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Store the mapping between numeric values and original fertilizer names
fertilizer_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("\nFertilizer name encoding mapping:")
for idx, name in fertilizer_mapping.items():
    print(f"{idx}: {name}")

# Create train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Also create encoded versions for models that require numeric targets
_, _, y_train_final_encoded, y_val_encoded = train_test_split(
    X_train_processed, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print("Training set shape:", X_train_final.shape)
print("Validation set shape:", X_val.shape)


# Create potentially useful features based on domain knowledge

# Calculate NPK ratio (Nitrogen:Phosphorous:Potassium)
def add_engineered_features(df):
    # Copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # NPK sum
    df_new['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    
    # NPK ratios
    df_new['N_to_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)  # Avoid division by zero
    df_new['N_to_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df_new['P_to_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    # Interaction features
    df_new['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    df_new['Temp_Moisture'] = df['Temparature'] * df['Moisture']
    df_new['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
    
    return df_new

# Apply feature engineering
X_train_final_eng = add_engineered_features(X_train_final)
X_val_eng = add_engineered_features(X_val)
X_test_processed_eng = add_engineered_features(X_test_processed)

print("Training features after engineering:", X_train_final_eng.shape)
print("Test features after engineering:", X_test_processed_eng.shape)
print("\nNew features:")
display(X_train_final_eng.iloc[:5, -7:])  # Show the new features


# Define multiple models to try
models = {
    # 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost GPU': XGBoost.XGBClassifier(n_estimators=1000, random_state=42, tree_method='gpu_hist'),
    'LightGBM GPU': lgb.LGBMClassifier(n_estimators=1000, random_state=42, device='gpu')
}

# Train models and collect validation accuracies
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_final_eng, y_train_final_encoded)
    
    # Predict on validation set
    val_pred = model.predict(X_val_eng)
    val_accuracy = (val_pred == y_val_encoded).mean()
    
    # Get prediction probabilities for MAP@3 calculation later
    val_proba = model.predict_proba(X_val_eng)
    
    results[name] = {
        'model': model,
        'accuracy': val_accuracy,
        'val_proba': val_proba
    }
    
    print(f"{name} validation accuracy: {val_accuracy:.4f}")
    print("-" * 40)

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")


# Implement Mean Average Precision @ 3 (MAP@3)
def map_at_3(y_true, y_pred_proba, classes):
    """
    Calculate Mean Average Precision @ 3 for multi-class classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for each class
        classes: List of class names
    
    Returns:
        MAP@3 score
    """
    # Convert true labels to indices if they're strings
    if isinstance(y_true.iloc[0], str):
        label_to_idx = {label: i for i, label in enumerate(classes)}
        y_true_idx = y_true.map(label_to_idx).values
    else:
        y_true_idx = y_true.values
    
    # Get top 3 predictions for each sample
    y_pred_idx = np.argsort(-y_pred_proba, axis=1)[:, :3]
    
    # Calculate AP@3 for each sample
    aps = []
    for i in range(len(y_true_idx)):
        ap = 0.0
        hits = 0.0
        
        for j, pred_idx in enumerate(y_pred_idx[i]):
            if pred_idx == y_true_idx[i]:
                hits += 1
                ap += hits / (j + 1)
                break
        
        if hits > 0:
            ap /= min(1, 3)  # Normalize by min(num_predictions, k=3)
        aps.append(ap)
    
    return np.mean(aps)

# Evaluate models using MAP@3
class_list = sorted(y_train.unique())
map_scores = {}

for name, result in results.items():
    val_proba = result['val_proba']
    map_score = map_at_3(y_val, val_proba, class_list)
    map_scores[name] = map_score
    print(f"{name} - MAP@3: {map_score:.4f}")

# Find the best model based on MAP@3
best_model_map = max(map_scores, key=map_scores.get)
print(f"\nBest model by MAP@3: {best_model_map} with score: {map_scores[best_model_map]:.4f}")

# Use the best model for further analysis
best_model = results[best_model_map]['model']


# Feature importance analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    feature_names = X_train_final_eng.columns
    
    # Create a dataframe for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title(f'Top 20 Features Importance for {best_model_map}')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 most important features:")
    display(importance_df.head(10))


print(y_train)


# Train the best model on the entire training set
print(f"Training the best model ({best_model_map}) on the entire training set...")
best_model.fit(X_train_final_eng, y_train_final_encoded)  # Use encoded target for training

# Predict probabilities for test set
test_proba = best_model.predict_proba(X_test_processed_eng)

# Get top 3 predictions for each test instance
top3_idx = np.argsort(-test_proba, axis=1)[:, :3]

# Convert indices back to original fertilizer names
top3_classes = np.array([
    [fertilizer_mapping[idx] for idx in row] 
    for row in top3_idx
])

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_classes]
})

print("Submission file preview:")
display(submission.head())

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

