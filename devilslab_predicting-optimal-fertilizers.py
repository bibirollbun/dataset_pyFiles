#Importing the important libraries 

import numpy as np
import pandas as pd

#importing libraries for visualization part
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report
import optuna

import itertools
import warnings
warnings.filterwarnings("ignore")




def inspect_dataset(filepath:str,dataset_name='Dataset'):
    """
    we are going to load and inspect the dataset, 
    displaying key information

    Parameters:
        -file_path(str):Path to dataset file
        -dataset_name(str): Name of the dataset to display
    """
    #Load the dataset
    try:
        df=pd.read_csv(filepath,index_col='id')
        print(f"\n==={dataset_name} Inspection ===")

        #Display the rows
        print(f"\n{dataset_name} - First 5 rows: ")
        print(df.head(5))

        #Display Shape
        print(f"\n{dataset_name} Shape: {df.shape}")

        #display dataset info
        print(f"\n{dataset_name} Info: :")
        print(df.info())

        #display basic statistics
        print(f"\n{dataset_name} Description:")
        print(df.describe())


        #check for missing values
        print(f"\n{dataset_name} Missing Values:")
        print(df.isnull().sum())

        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.Please recheck the file path.")
        return None
    except Exception as e:
        print(f"Error Loading {dataset_name} : {str(e)}")
        return None


#Load and inspect Training data
train_path='/kaggle/input/playground-series-s5e6/train.csv'
train_df=inspect_dataset(train_path,dataset_name='Training Data')

#Load and inspect Testing data
test_path='/kaggle/input/playground-series-s5e6/test.csv'
test_df=inspect_dataset(test_path,dataset_name='Testing Data')


#Target distribution graph
target_clmn=train_df['Fertilizer Name']
print(f"Training Data Target Distribution ({target_clmn}):")
print(target_clmn.value_counts(normalize=True))
#visualize the target distribution
plt.figure(figsize=(8,5))
sns.countplot(x=target_clmn,data=train_df)
plt.title("Training Data - Target Distribution")
plt.show()


#perform EDA
def perform_eda(df,dataset_name='Dataset',target_column='Target',numerical_cols=None,categorical_cols=None):
    """
    perform Exploratory Data Analysis on the Dataset , Visualizing key patterns.

    Parameters:
        - df(pd.DataFrame):Dataset to analyze
        - dataset_name(str):Name of the dataset to display
        - target_column(str):Name of the target column
        - numerical_cols(list,optional):List of numerical column names
        - Categorical_cols(list,optional): List of categorical columns
        
    """
    print(f"\n==={dataset_name} EDA ===")
    # If numerical/categorical columns not provided, infer them
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 1. Target Variable Analysis (for classification)
    if target_column in df.columns:
        print(f"\nTarget Variable ({target_column}) Analysis:")
        print(df[target_column].value_counts(normalize=True))
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=target_column, data=df, palette='viridis')
        plt.title(f'{dataset_name} - {target_column} Distribution')
        plt.xticks(rotation=45)
        plt.show()

    # 2. Numerical Features Analysis
    if numerical_cols:
        print(f"\nNumerical Features Analysis ({len(numerical_cols)} columns):")
        
        # Distribution of numerical features
        for col in numerical_cols[:5]:  # Limit to first 5 for brevity
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True, color='blue')
            plt.title(f'{dataset_name} - Distribution of {col}')
            plt.show()

        # Correlation matrix for numerical features
        plt.figure(figsize=(12, 8))
        corr = df[numerical_cols + [target_column] if target_column in df.columns and df[target_column].dtype in ['int64', 'float64'] else numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f'{dataset_name} - Correlation Matrix')
        plt.show()

    # 3. Categorical Features Analysis
    if categorical_cols:
        print(f"\nCategorical Features Analysis ({len(categorical_cols)} columns):")
        
        # Count plots for categorical features
        for col in categorical_cols[:5]:  
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, data=df, palette='muted')
            plt.title(f'{dataset_name} - Distribution of {col}')
            plt.xticks(rotation=45)
            plt.show()

        # Relationship with target (if target exists)
        if target_column in df.columns:
            for col in categorical_cols[:3]: 
                plt.figure(figsize=(10, 6))
                sns.countplot(x=col, hue=target_column, data=df, palette='Set2')
                plt.title(f'{dataset_name} - {col} vs {target_column}')
                plt.xticks(rotation=45)
                plt.show()

    # 4. Outlier Detection (for numerical features)
    if numerical_cols:
        print(f"\nOutlier Detection for Numerical Features:")
        for col in numerical_cols[:5]:  
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3).sum()
            print(f"{col}: {outliers} outliers (z-score > 3)")

    # 5. Missing Values Heatmap (if any missing values)
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title(f'{dataset_name} - Missing Values Heatmap')
        plt.show()
    


# Perform EDA on training data
perform_eda(train_df, dataset_name='Training Data', target_column='Fertilizer Name')


train_df.info()


def preprocess_data(train_df, test_df, target_column='Fertilizer Name'):
    """
    Preprocess train and test datasets, automatically detecting numerical and categorical features.

    Parameters:
        - train_df (pd.DataFrame): Training dataset.
        - test_df (pd.DataFrame): Test dataset.
        - target_column (str): Name of the target column in train_df.

    Returns:
        - X_train (pd.DataFrame): Preprocessed training features.
        - X_test (pd.DataFrame): Preprocessed test features.
        - y_train (np.array): Encoded target variable.
        - preprocessor (ColumnTransformer): Fitted preprocessor for reuse.
        - label_encoder (LabelEncoder): Fitted label encoder for target.
    """
    # Print column names for debugging
    print(f"Train Data Columns: {train_df.columns.tolist()}")
    print(f"Test Data Columns: {test_df.columns.tolist()}")

    # Validate target column in train_df
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train_df. Available columns: {train_df.columns.tolist()}")

    # Automatically detect numerical and categorical columns
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Validate feature columns exist in both train and test
    feature_cols = numerical_cols + categorical_cols
    missing_train_cols = [col for col in feature_cols if col not in train_df.columns]
    missing_test_cols = [col for col in feature_cols if col not in test_df.columns]
    if missing_train_cols or missing_test_cols:
        raise ValueError(f"Missing columns in train: {missing_train_cols}, test: {missing_test_cols}")

    print(f"Detected Numerical Columns: {numerical_cols}")
    print(f"Detected Categorical Columns: {categorical_cols}")

    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.copy()  # Test data has no target

    # Encode target variable (for classification)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    print(f"Target Classes: {list(label_encoder.classes_)}")

    # Define preprocessing for numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('scaler', MinMaxScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ])

    # Fit and transform training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    feature_names = (
        numerical_cols +
        list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols))
    )

    # Convert to DataFrame for easier handling
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=X_test.index)

    print(f"\nPreprocessed Training Data Shape: {X_train_preprocessed.shape}")
    print(f"Preprocessed Test Data Shape: {X_test_preprocessed.shape}")
    print("\nPreprocessed Training Data Sample:")
    print(X_train_preprocessed.head())

    return X_train_preprocessed, X_test_preprocessed, y_train_encoded, preprocessor, label_encoder


# Preprocess the data
X_train, X_test, y_train, preprocessor, label_encoder = preprocess_data(
    train_df, test_df, target_column='Fertilizer Name'
)

# Optional: Inspect preprocessed test data
print("\nPreprocessed Test Data Sample:")
print(X_test.head())


def train_and_evaluate_models(X_train, y_train, X_test, label_encoder, k_folds=5):
    """
    Train and evaluate multiple models with KFold cross-validation and Optuna tuning.

    Parameters:
        - X_train (pd.DataFrame): Preprocessed training features.
        - y_train (np.array): Encoded target variable.
        - X_test (pd.DataFrame): Preprocessed test features.
        - label_encoder (LabelEncoder): Fitted label encoder for decoding predictions.
        - k_folds (int): Number of folds for cross-validation.

    Returns:
        - models (dict): Trained models.
        - cv_scores (dict): Cross-validation accuracy scores.
        - test_predictions (dict): Predictions on test set.
    """
    # Initialize models
    models = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    cv_scores = {}
    test_predictions = {}
    
    # KFold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Cross-validation
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Fit model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict and evaluate
            y_pred = model.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(accuracy)
            print(f"{model_name} - Fold {fold+1} Accuracy: {accuracy:.4f}")
        
        # Store mean CV score
        cv_scores[model_name] = np.mean(fold_scores)
        print(f"{model_name} - Mean CV Accuracy: {cv_scores[model_name]:.4f} ± {np.std(fold_scores):.4f}")
        
        # Train on full training data
        model.fit(X_train, y_train)
        
        # Predict on test set
        test_pred = model.predict(X_test)
        test_predictions[model_name] = label_encoder.inverse_transform(test_pred)  # Decode to original classes
    
    # Tune best model with Optuna (e.g., XGBoost)
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        xgb = XGBClassifier(**param, random_state=42, eval_metric='mlogloss')
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]
            xgb.fit(X_train_fold, y_train_fold)
            y_pred = xgb.predict(X_val_fold)
            scores.append(accuracy_score(y_val_fold, y_pred))
        return np.mean(scores)
    
    print("\nTuning XGBoost with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Adjust n_trials for better tuning
    best_params = study.best_params
    print(f"Best XGBoost Parameters: {best_params}")
    
    # Train tuned XGBoost
    tuned_xgb = XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss')
    tuned_xgb.fit(X_train, y_train)
    test_predictions['Tuned_XGBoost'] = label_encoder.inverse_transform(tuned_xgb.predict(X_test))
    
    # Evaluate tuned model
    cv_scores['Tuned_XGBoost'] = study.best_value
    print(f"Tuned XGBoost - Mean CV Accuracy: {cv_scores['Tuned_XGBoost']:.4f}")
    
    return models, cv_scores, test_predictions


# Train and evaluate models
models, cv_scores, test_predictions = train_and_evaluate_models(
    X_train, y_train, X_test, label_encoder, k_folds=5
)

# Print CV scores
print("\nCross-Validation Scores:")
for model_name, score in cv_scores.items():
    print(f"{model_name}: {score:.4f}")

# Save predictions for submission (using best model, e.g., Tuned_XGBoost)
submission = pd.DataFrame({
    'id': test_df.index,  # From Step 2, 'id' is the index
    'Fertilizer Name': test_predictions['Tuned_XGBoost']
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")


def create_submission(test_df, test_predictions, model_name='Tuned_XGBoost', submission_file='submission.csv'):
    """
    Create a Kaggle submission file from test predictions.

    Parameters:
        - test_df (pd.DataFrame): Test dataset with 'id' as index.
        - test_predictions (dict): Dictionary of model predictions from train_and_evaluate_models.
        - model_name (str): Name of the model to use for submission (e.g., 'Tuned_XGBoost').
        - submission_file (str): Output file name for submission.

    Returns:
        - None (saves submission.csv).
    """
    if model_name not in test_predictions:
        raise ValueError(f"Model '{model_name}' not found in predictions. Available: {list(test_predictions.keys())}")

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df.index,  # Assumes 'id' is the index from Step 2
        'Fertilizer Name': test_predictions[model_name]
    })

    # Save to CSV
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    print(submission.head())


# Create submission file
create_submission(test_df, test_predictions, model_name='Tuned_XGBoost', submission_file='submission.csv')

