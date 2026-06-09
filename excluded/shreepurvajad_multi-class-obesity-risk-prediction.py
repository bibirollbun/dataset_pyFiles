import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')


print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())


print("Dataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


# Statistical summary
print("Statistical Summary:")
display(df.describe())


print(df.columns.tolist())



# Target variable analysis
print("Target Variable Distribution:")
target_counts = df['NObeyesdad'].value_counts()
print(target_counts)

plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='NObeyesdad', order=target_counts.index)
plt.title('Distribution of Target Variable (NObeyesdad)')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


# Numerical features distribution
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()


# Categorical features analysis
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    if i < len(axes):
        df[col].value_counts().plot(kind='bar', ax=axes[i], title=col)
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Correlation analysis
plt.figure(figsize=(12, 10))
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()


# Relationship between Weight, Height and Target
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Height', y='Weight', hue='NObeyesdad', alpha=0.7, s=60)
plt.title('Height vs Weight colored by Obesity Category')
plt.tight_layout()
plt.show()


# Age distribution by obesity category
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='NObeyesdad', y='Age')
plt.xticks(rotation=45)
plt.title('Age Distribution across Obesity Categories')
plt.tight_layout()
plt.show()


# Create a copy for feature engineering
df_processed = df.copy()

# 1. Create BMI feature - Most important feature for obesity classification
df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)

# 2. Create age groups
df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                 bins=[0, 18, 30, 45, 60, 100], 
                                 labels=['Teen', 'Young', 'Adult', 'Middle', 'Senior'])

# 3. Create weight status based on BMI (simplified)
def get_weight_status(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df_processed['Weight_Status'] = df_processed['BMI'].apply(get_weight_status)

# 4. Create interaction features
df_processed['Age_Weight_Interaction'] = df_processed['Age'] * df_processed['Weight']
df_processed['Family_History_Weight'] = df_processed['family_history_with_overweight'].map({'yes': 1, 'no': 0}) * df_processed['Weight']

# 5. Create lifestyle score
df_processed['Lifestyle_Score'] = (
    df_processed['FCVC'] +  # Frequency of vegetable consumption
    df_processed['NCP'] +   # Number of main meals
    df_processed['FAF'] +   # Physical activity frequency
    (3 - df_processed['TUE'])  # Inverse of technology usage
)

# 6. Create health risk indicator
df_processed['Health_Risk_Indicator'] = (
    (df_processed['family_history_with_overweight'] == 'yes').astype(int) +
    (df_processed['FAVC'] == 'no').astype(int) +  # No high cal food consumption
    (df_processed['SMOKE'] == 'yes').astype(int)  # Smoking
)

print("New features created:")
print(df_processed[['BMI', 'Age_Group', 'Weight_Status', 'Lifestyle_Score', 'Health_Risk_Indicator']].head())


label_encoders = {}

# Binary categorical variables
binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Multi-class categorical variables
multi_cat_cols = ['CAEC', 'CALC', 'MTRANS', 'Age_Group', 'Weight_Status']
for col in multi_cat_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
df_processed['NObeyesdad_encoded'] = target_encoder.fit_transform(df_processed['NObeyesdad'])

print("Categorical encoding completed.")
print(f"Target classes: {target_encoder.classes_}")


# Prepare features for modeling
# Select features for modeling
feature_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
    'CALC', 'MTRANS', 'BMI', 'Age_Group', 'Weight_Status', 
    'Age_Weight_Interaction', 'Family_History_Weight', 'Lifestyle_Score',
    'Health_Risk_Indicator'
]

X = df_processed[feature_columns]
y = df_processed['NObeyesdad_encoded']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


print("Training XGBoost Classifier...")

# Basic XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predict on validation set
y_pred_xgb = xgb_model.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)

print(f"XGBoost Baseline Accuracy: {xgb_accuracy:.4f}")


print("Performing Hyperparameter Tuning for XGBoost...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# Using RandomizedSearchCV for faster tuning
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    xgb_model, 
    param_grid, 
    n_iter=20, 
    cv=3, 
    scoring='accuracy', 
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: {:.4f}".format(random_search.best_score_))

# Get the best model
best_xgb_model = random_search.best_estimator_


y_pred_best_xgb = best_xgb_model.predict(X_val)
final_accuracy = accuracy_score(y_val, y_pred_best_xgb)

print(f"Final XGBoost Model Accuracy: {final_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_best_xgb, target_names=target_encoder.classes_))


# Confusion Matrix for XGBoost
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, y_pred_best_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=target_encoder.classes_, 
           yticklabels=target_encoder.classes_)
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('XGBoost - Top 15 Feature Importances')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
display(feature_importance.head(10))


def preprocess_test_data(test_df, label_encoders, feature_columns):
    """
    Preprocess test data using the same transformations as training data
    """
    test_processed = test_df.copy()
    
    # Create only the engineered features that we actually used in training
    if 'BMI' in feature_columns:
        test_processed['BMI'] = test_processed['Weight'] / (test_processed['Height'] ** 2)
    
    if 'Age_Group' in feature_columns:
        test_processed['Age_Group'] = pd.cut(test_processed['Age'], 
                                          bins=[0, 18, 30, 45, 60, 100], 
                                          labels=['Teen', 'Young', 'Adult', 'Middle', 'Senior'])
    
    if 'Weight_Status' in feature_columns:
        test_processed['Weight_Status'] = test_processed['BMI'].apply(get_weight_status)
    
    if 'Age_Weight_Interaction' in feature_columns:
        test_processed['Age_Weight_Interaction'] = test_processed['Age'] * test_processed['Weight']
    
    if 'Family_History_Weight' in feature_columns:
        test_processed['Family_History_Weight'] = test_processed['family_history_with_overweight'].map({'yes': 1, 'no': 0}) * test_processed['Weight']
    
    if 'Lifestyle_Score' in feature_columns:
        test_processed['Lifestyle_Score'] = (
            test_processed['FCVC'] + 
            test_processed['NCP'] + 
            test_processed['FAF'] + 
            (3 - test_processed['TUE'])
        )
    
    if 'Health_Risk_Indicator' in feature_columns:
        test_processed['Health_Risk_Indicator'] = (
            (test_processed['family_history_with_overweight'] == 'yes').astype(int) +
            (test_processed['FAVC'] == 'no').astype(int) +
            (test_processed['SMOKE'] == 'yes').astype(int)
        )
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in test_processed.columns:
            # Handle unseen labels
            test_processed[col] = test_processed[col].astype(str)
            unknown_mask = ~test_processed[col].isin(encoder.classes_)
            if unknown_mask.any():
                # Replace unknown with most frequent
                test_processed.loc[unknown_mask, col] = encoder.classes_[0]
            test_processed[col] = encoder.transform(test_processed[col])
    
    # Select only the features that exist
    X_test = test_processed[feature_columns]
    
    return X_test

print("Updated preprocessing function created.")


# Load and process test data
print("Loading test data...")
test_df = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

print(f"Test data shape: {test_df.shape}")



print("Test data columns:", test_df.columns.tolist())



display(test_df.head())


# Now preprocess test data with the correct features
print("Preprocessing test data with correct features...")
X_test_processed = preprocess_test_data(test_df, label_encoders, feature_columns)

print(f"Processed test data shape: {X_test_processed.shape}")
print("Success! Test data preprocessing completed.")


# Fit the model first
xgb_model.fit(X_train, y_train)

# Then make predictions
test_predictions = xgb_model.predict(X_test_processed)


# Make predictions on test data
print("Making predictions on test data...")
test_predictions = xgb_model.predict(X_test_processed)
test_predicted_classes = target_encoder.inverse_transform(test_predictions)

print("Predictions completed!")
print("Sample predictions:", test_predicted_classes[:10])


submission = pd.DataFrame({
    'id': test_df['id'],
    'NObeyesdad': test_predicted_classes
})

print("Submission file preview:")
display(submission.head())
print(f"Submission file shape: {submission.shape}")

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


prediction_counts = submission['NObeyesdad'].value_counts()
print("Prediction distribution in test set:")
print(prediction_counts)

plt.figure(figsize=(12, 6))
sns.countplot(data=submission, y='NObeyesdad', order=prediction_counts.index)
plt.title('Distribution of Predictions in Test Set')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


import joblib

model_artifacts = {
    'model': xgb_model,
    'label_encoders': label_encoders,
    'target_encoder': target_encoder,
    'feature_columns': feature_columns,
    'preprocess_function': preprocess_test_data
}

joblib.dump(model_artifacts, 'xgboost_obesity_model.pkl')
print("Model artifacts saved successfully as 'xgboost_obesity_model.pkl'")


# Calculate validation accuracy first
y_val_pred = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

# Now run your final summary
print("="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Model: XGBoost Classifier")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Number of Features: {len(feature_columns)}")
print(f"Features used: {feature_columns}")
print(f"Test Predictions: {len(submission)}")
print(f"Submission File: 'submission.csv'")
print("="*50)

