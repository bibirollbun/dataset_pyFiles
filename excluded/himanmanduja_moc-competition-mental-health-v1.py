# MOC Competition: Mental Health - Advanced EDA

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import joblib
import warnings

# Suppress FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)


# Set plot style
sns.set_style("whitegrid")

# 2. Load Data
try:
    df = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the dataset is in the correct directory.")
    exit()


# 3. Initial Data Inspection
print("\n--- Initial Data Inspection ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDropping 'id' and 'Name' columns as they are not needed for modeling.")
df.drop(columns=['id', 'Name'], inplace=True)


# 4. Null Value Analysis
print("\n--- Null Value Analysis ---")
null_values = df.isnull().sum()
null_percentage = (null_values / len(df)) * 100
null_df = pd.DataFrame({'Null Count': null_values, 'Null Percentage': null_percentage})
null_df = null_df[null_df['Null Count'] > 0].sort_values(by='Null Count', ascending=False)
print("Columns with null values:")
print(null_df)

# Visualize missing values heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap', fontsize=16)
plt.show()


# 5. Univariate Analysis
print("\n--- Univariate Analysis ---")

# Target Variable: Depression
plt.figure(figsize=(7, 5))
sns.countplot(x='Depression', data=df, palette='pastel')
plt.title('Distribution of Depression', fontsize=16)
plt.xlabel('Depression (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Categorical Features
categorical_features = ['Gender', 'Working Professional or Student', 'Sleep Duration', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=feature, data=df, order=df[feature].value_counts().index, palette='viridis')
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.show()


# 6. Bivariate Analysis
print("\n--- Bivariate Analysis: Feature vs. Depression ---")

# Categorical Features vs. Depression
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Depression', data=df, palette='magma')
    plt.title(f'Depression by {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Depression', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

# Numerical Features vs. Depression
numerical_features = ['Age', 'Financial Stress', 'Work/Study Hours']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Depression', y=feature, data=df, palette='coolwarm')
    plt.title(f'Depression by {feature}', fontsize=16)
    plt.xlabel('Depression (0: No, 1: Yes)', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.show()



# 7. Correlation Analysis
print("\n--- Correlation Analysis ---")
# Create a copy for encoding
df_corr = df.copy()

# Label encode object columns for correlation matrix
for col in df_corr.select_dtypes(include=['object']).columns:
    df_corr[col] = df_corr[col].astype('category').cat.codes

# Impute NaNs with median for correlation matrix calculation
for col in df_corr.columns:
    if df_corr[col].isnull().any():
        df_corr[col] = df_corr[col].fillna(df_corr[col].median())

# Correlation Matrix
plt.figure(figsize=(18, 14))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of All Features', fontsize=20)
plt.show()

print("\nEDA notebook complete. ðŸ“ˆ")


def preprocess_data(df, encoders=None, is_training=False):
    """
    Preprocesses data for training or prediction.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        encoders (dict): A dictionary of fitted LabelEncoders. Required if is_training=False.
        is_training (bool): If True, fits new encoders. If False, uses provided encoders.
        
    Returns:
        pd.DataFrame: The preprocessed dataframe.
        dict: The fitted label encoders (only if is_training=True).
    """
    print(f"Preprocessing data... Mode: {'Training' if is_training else 'Prediction'}")
    
    processed_df = df.copy()
    
    # --- Step 1: Handle Missing Values ---
    for col in ['Dietary Habits', 'Degree', 'Financial Stress']:
        if processed_df[col].isnull().any():
            mode_val = processed_df[col].mode()[0]
            processed_df[col].fillna(mode_val, inplace=True)

    for col in ['Academic Pressure', 'CGPA', 'Study Satisfaction']:
        processed_df[f'{col}_is_missing'] = processed_df[col].isnull().astype(int)
        processed_df[col].fillna(0, inplace=True)

    for col in ['Work Pressure', 'Job Satisfaction']:
        processed_df[f'{col}_is_missing'] = processed_df[col].isnull().astype(int)
        processed_df[col].fillna(0, inplace=True)
        
    processed_df['Profession'].fillna('Not Applicable', inplace=True)

    # --- Step 2: Encode Categorical Variables ---
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    
    if is_training:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            encoders[col] = { 'classes': le.classes_.tolist(), 'unknown': -1 }
        print("Fitted and saved new label encoders.")
        return processed_df, encoders
    else:
        if encoders is None:
            raise ValueError("Encoders must be provided for prediction mode.")
        for col in categorical_cols:
            le = LabelEncoder()
            le.classes_ = np.array(encoders[col]['classes'])
            # Handle unseen values in test data by mapping them to a special 'unknown' value
            processed_df[col] = processed_df[col].map(lambda s: s if s in le.classes_ else '<unknown>')
            # Add '<unknown>' to the classes if it's not there
            if '<unknown>' not in le.classes_:
                 le.classes_ = np.append(le.classes_, '<unknown>')
            processed_df[col] = le.transform(processed_df[col])
        print("Applied saved label encoders.")
        return processed_df



def train_and_save_model():
    """
    Loads training data, preprocesses it, evaluates the model, 
    then retrains on all data and saves the model and encoders.
    """
    print("--- Starting Model Training and Evaluation ---")
    # Load data
    df_train = pd.read_csv('/kaggle/input/moc-competition-mental-health/train.csv')
    
    # Preprocess training data
    X = df_train.drop(['id', 'Name', 'Depression'], axis=1)
    y = df_train['Depression']
    X_processed, encoders = preprocess_data(X, is_training=True)

    # Save encoders to a JSON file
    with open('label_encoders.json', 'w') as f:
        json.dump(encoders, f, indent=4)
    print("Label encoders saved to 'label_encoders.json'")

    # --- Model Evaluation Step ---
    print("\n--- Evaluating Model Performance on a Validation Set ---")
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Initialize the XGBoost model
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train on the training subset
    xgb_classifier.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred_val = xgb_classifier.predict(X_val)
    
    # Show reports
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val, target_names=['Not Depressed', 'Depressed']))
    
    # Show confusion matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Depressed', 'Depressed'], 
                yticklabels=['Not Depressed', 'Depressed'])
    plt.title('Validation Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()

    # --- Final Model Training ---
    print("\n--- Retraining Model on Full Dataset for Saving ---")
    # Re-initialize and train on the entire dataset to build the most robust model
    full_data_scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=full_data_scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X_processed, y)
    
    # Save the final trained model
    joblib.dump(final_model, 'xgboost_model.joblib')
    print("Final model saved to 'xgboost_model.joblib'")
    print("--- Training Complete ---")



# --- PART 3: PREDICTOR CLASS FOR USER INPUT ---

class MentalHealthPredictor:
    def __init__(self, model_path='xgboost_model.joblib', encoders_path='label_encoders.json'):
        """
        Initializes the predictor by loading the model and encoders.
        """
        print("--- Initializing Predictor ---")
        try:
            self.model = joblib.load(model_path)
            with open(encoders_path, 'r') as f:
                self.encoders = json.load(f)
            print("Model and encoders loaded successfully.")
        except FileNotFoundError:
            print("Error: Model or encoder file not found. Please run the training script first.")
            self.model = None
            self.encoders = None
        print("--- Predictor Ready ---")

    def predict_single(self, user_data):
        """
        Predicts the mental health status for a single user's data.
        
        Args:
            user_data (dict): A dictionary containing user features.
            
        Returns:
            str: The prediction ('Depressed' or 'Not Depressed').
        """
        if not self.model:
            return "Predictor not initialized."
            
        # Convert dictionary to a DataFrame with a single row
        df_user = pd.DataFrame([user_data])
        
        # Preprocess the user data using saved encoders
        df_user_processed = preprocess_data(df_user, encoders=self.encoders, is_training=False)
        
        # Ensure column order matches the training data
        # This is a safety check in case the dictionary order is different
        training_cols = self.model.get_booster().feature_names
        df_user_processed = df_user_processed[training_cols]

        # Make prediction
        prediction = self.model.predict(df_user_processed)[0]
        
        return "Depressed" if prediction == 1 else "Not Depressed"


# --- PART 4: BATCH PREDICTION AND SUBMISSION FILE CREATION ---

def create_submission_file():
    """
    Loads the test data, predicts outcomes, and creates a submission.csv file.
    """
    print("\n--- Creating Submission File ---")
    # Load test data and encoders
    df_test = pd.read_csv('/kaggle/input/moc-competition-mental-health/test.csv')
    test_ids = df_test['id']
    X_test = df_test.drop(['id', 'Name'], axis=1)

    with open('label_encoders.json', 'r') as f:
        encoders = json.load(f)
        
    # Load model
    model = joblib.load('xgboost_model.joblib')
    
    # Preprocess test data
    X_test_processed = preprocess_data(X_test, encoders=encoders, is_training=False)
    
    # Ensure column order matches training data
    training_cols = model.get_booster().feature_names
    X_test_processed = X_test_processed[training_cols]

    # Make predictions
    predictions = model.predict(X_test_processed)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({'id': test_ids, 'Depression': predictions})
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")
    print("Top 5 rows of submission file:")
    print(submission_df.head())
    print("--- Submission Complete ---")
# --- MAIN EXECUTION ---


# Step 1: Train and save the model and encoders
train_and_save_model()


# Step 2: Create the submission file for the competition
create_submission_file()


# Step 3: Demonstrate the predictor class with a sample user
print("\n--- Demonstrating Single Prediction ---")
predictor = MentalHealthPredictor()

# Example user data (a student with high academic pressure)
sample_user = {
    'Gender': 'Female',
    'Age': 21.0,
    'City': 'Delhi',
    'Working Professional or Student': 'Student',
    'Profession': np.nan, # Student has no profession
    'Academic Pressure': 5.0,
    'Work Pressure': np.nan,
    'CGPA': 8.5,
    'Study Satisfaction': 2.0,
    'Job Satisfaction': np.nan,
    'Sleep Duration': 'Less than 5 hours',
    'Dietary Habits': 'Unhealthy',
    'Degree': 'B.Tech',
    'Have you ever had suicidal thoughts ?': 'Yes',
    'Work/Study Hours': 10.0,
    'Financial Stress': 4.0,
    'Family History of Mental Illness': 'Yes'
}

if predictor.model:
  prediction_result = predictor.predict_single(sample_user)
  print(f"\nPrediction for sample user: {prediction_result}")




