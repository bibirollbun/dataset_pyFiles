# Force reinstall specific versions of required libraries to avoid compatibility issues
!pip install --force-reinstall scikit-learn==1.4.2 imbalanced-learn==0.12.0 pandas==2.2.2 numpy==1.26.4 matplotlib==3.7.5 seaborn==0.13.2


# Verify library versions
import sklearn
import imblearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
print("scikit-learn version:", sklearn.__version__)
print("imbalanced-learn version:", imblearn.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("matplotlib version:", matplotlib.__version__)
print("seaborn version:", sns.__version__)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Load the training and test datasets
train_data = pd.read_csv("/kaggle/input/spam-emails12345/train.csv")
test_data = pd.read_csv("/kaggle/input/spam-emails12345/test.csv")


# Display basic information about the dataset
print("Training Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("\nTraining Data Columns:", train_data.columns)


# Check for missing values
print("\nMissing Values in Training Set:\n", train_data.isnull().sum())
print("\nMissing Values in Test Set:\n", test_data.isnull().sum())


# Explore the target variable distribution
print("\nTarget Variable Distribution:\n", train_data['Exited'].value_counts(normalize=True) * 100)

# Visualize the target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Exited', data=train_data)
plt.title('Distribution of Target Variable (Exited)')
plt.xlabel('Exited (0: No, 1: Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('target_distribution.png')



# Define features and target
X = train_data.drop(columns=['Exited', 'id', 'CustomerId', 'Surname'])
y = train_data['Exited']


# Split the data into training and validation sets (stratified split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Define categorical and numerical columns
categorical_cols = ['Geography', 'Gender']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])



# Apply preprocessing to training and validation sets
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)


# Train an initial RandomForestClassifier on the imbalanced data
initial_model = RandomForestClassifier(random_state=42)
initial_model.fit(X_train_preprocessed, y_train)


# Predict probabilities and labels on the validation set
y_val_pred_proba = initial_model.predict_proba(X_val_preprocessed)[:, 1]
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)


# Evaluate the initial model
print("Initial ROC-AUC:", roc_auc_score(y_val, y_val_pred_proba))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))


# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)


# Check the new class distribution
print("\nNew Target Distribution After SMOTE:\n", pd.Series(y_train_resampled).value_counts(normalize=True) * 100)


# Retrain the RandomForestClassifier on the augmented data
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train_resampled, y_train_resampled)


# Predict probabilities and labels on the validation set using the retrained model
y_val_pred_proba_final = final_model.predict_proba(X_val_preprocessed)[:, 1]
y_val_pred_final = (y_val_pred_proba_final >= 0.5).astype(int)


# Evaluate the final model
print("Final ROC-AUC:", roc_auc_score(y_val, y_val_pred_proba_final))
print("\nFinal Classification Report:\n", classification_report(y_val, y_val_pred_final))


# Preprocess the test data
test_data_processed = test_data.drop(columns=['id', 'CustomerId', 'Surname'])
X_test_preprocessed = preprocessor.transform(test_data_processed)


# Make predictions using the retrained model
test_pred_proba = final_model.predict_proba(X_test_preprocessed)[:, 1]


# Prepare the submission file
submission_df = pd.DataFrame({'id': test_data['id'], 'Exited': test_pred_proba})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: 'submission.csv'")

