!pip install -q scikit-learn==1.2.2 imbalanced-learn==0.10.1


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint as sp_randint, uniform
import warnings
import os

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Configure warnings and display settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("✓ Libraries imported successfully")


# Load dataset
data = pd.read_csv("/kaggle/input/bank-marketing-prediction/train.csv")

# Display basic information
print(f"Dataset Shape: {data.shape}")
print("\nFirst few rows:")
display(data.head())

print("\nBasic information:")
data.info()


# Check for missing values
print("Missing Values:")
print(data.isnull().sum())
print(f"\nTotal missing: {data.isnull().sum().sum()}")

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Check for 'unknown' values in categorical columns
categorical_cols = data.select_dtypes(include='object').columns

print("\nUnknown values in categorical features:")
for col in categorical_cols:
    unknown_count = (data[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"{col:15s}: {unknown_count:5d} ({unknown_count/len(data)*100:.2f}%)")


# Rename target variable for clarity
data.rename(columns={'y': 'subscribed'}, inplace=True)

# Plot target distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='subscribed', palette=['#FF6B6B', '#4ECDC4'])

# Add value labels on top of bars
for i in ax.containers:
    ax.bar_label(i, fmt='%d', padding=3)

# Style the plot
plt.title('Distribution of Target Variable (Term Deposit Subscription)', pad=20)
plt.xlabel('Subscription Status')
plt.ylabel('Number of Customers')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print target distribution percentages
print("\nTarget Distribution:")
target_dist = data['subscribed'].value_counts(normalize=True) * 100
print(target_dist)


# Get numerical columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('SampleId')  # Remove ID column

# Create distribution plots for numerical features
for col in ['age', 'balance', 'duration', 'campaign']:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original distribution
    sns.histplot(data=data, x=col, kde=True, ax=ax1, color='#4ECDC4')
    ax1.set_title(f'{col.capitalize()} Distribution', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Distribution by target
    sns.boxplot(data=data, x='subscribed', y=col, ax=ax2, palette=['#FF6B6B', '#4ECDC4'])
    ax2.set_title(f'{col.capitalize()} by Subscription Status', pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Check for outliers
print("Outlier Analysis:")
for col in ['age', 'balance', 'duration', 'campaign']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"{col:10s}: {outliers:5d} outliers ({outliers/len(data)*100:.2f}%)")


# Analyze categorical features
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize=(12, 5))
    
    # Create subplot for distribution
    plt.subplot(1, 2, 1)
    sns.countplot(data=data, y=col, order=data[col].value_counts().index, color='#4ECDC4')
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xlabel('Count')
    
    # Create subplot for subscription rate
    plt.subplot(1, 2, 2)
    subscription_rate = data.groupby(col)['subscribed'].mean().sort_values(ascending=True) * 100
    subscription_rate.plot(kind='barh', color='#FF6B6B')
    plt.title(f'Subscription Rate by {col.capitalize()} (%)')
    plt.xlabel('Subscription Rate (%)')
    
    plt.tight_layout()
    plt.show()


# Calculate correlation matrix for numerical features
corr = data.corr(numeric_only=True)

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', pad=20)
plt.tight_layout()
plt.show()

# Create focused correlation plot with target
target_corr = corr['subscribed'].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
target_corr.plot(kind='bar')
plt.title('Feature Correlations with Target (subscribed)', pad=20)
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate mutual information with target
X = data.drop(columns=['SampleId', 'subscribed'])
y = data['subscribed'].astype(int)

# Encode categorical features
for c in cat_cols:
    if c in X.columns:
        X[c] = pd.factorize(X[c])[0]

# Mark discrete features
discrete_mask = [True if c in cat_cols else False for c in X.columns]

# Compute mutual information
mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=74)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.title('Feature Importance by Mutual Information')
plt.xlabel('Mutual Information (bits)')
plt.tight_layout()
plt.show()


# Drop unnecessary features
data.drop(columns=['SampleId', 'previous', 'default', 'loan'], inplace=True)
print("Dropped low-value features")

# Age binning
data['age'] = pd.cut(data['age'], 
                    bins=[17, 29, 45, 60, 95], 
                    labels=['young', 'adult', 'middle_aged', 'senior'])
print("\nAge categories created")

# Log transformation for numerical features
for col in ['balance', 'duration', 'campaign']:
    if col == 'balance':
        balance_min = data[col].min()
        data[col] = np.log1p(data[col] - balance_min)
    else:
        data[col] = np.log1p(data[col])
    print(f"Log transformation applied to {col}")

# Cyclical encoding for month
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

data['month_num'] = data['month'].map(month_mapping)
data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12)
data.drop(columns=['month', 'month_num'], inplace=True)

print("\nCyclical encoding applied to month")

# Categorical encoding
# One-hot encoding for nominal variables
data = pd.get_dummies(data, columns=['job', 'marital', 'contact', 'poutcome'], drop_first=True)

# Ordinal encoding for education
data['education'] = data['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})

# Binary encoding for housing
data['housing'] = data['housing'].replace({'yes': 1, 'no': 0})

# Create age dummy variables
age_dummies = pd.get_dummies(data['age'], prefix='age', drop_first=True)
data = pd.concat([data, age_dummies], axis=1)
data.drop(columns=['age'], inplace=True)

print("\nCategorical encoding completed")

# Display final dataset info
print(f"\nFinal dataset shape: {data.shape}")
print(f"Features: {data.shape[1]}")
print(f"Samples: {data.shape[0]}")
print(f"Missing values: {data.isnull().sum().sum()}")


# Split features and target
X = data.drop(columns=['subscribed'])
y = data['subscribed']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=74, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=74, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("\nBalanced training set shape:", X_train_scaled.shape)
print("Original class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Balanced class distribution:", dict(zip(*np.unique(y_train_balanced, return_counts=True))))


# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred)
    }
    
    return results

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=74),
    'Random Forest': RandomForestClassifier(random_state=74),
    'XGBoost': xgb.XGBClassifier(random_state=74),
    'Gradient Boosting': GradientBoostingClassifier(random_state=74),
    'AdaBoost': AdaBoostClassifier(random_state=74),
    'Decision Tree': DecisionTreeClassifier(random_state=74),
    'KNN': KNeighborsClassifier(),
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    results[name] = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    
# Create results DataFrame
results_df = pd.DataFrame(results).round(4)
print("\nModel Comparison:")
display(results_df)


# Hyperparameter tuning for Random Forest
rf_param_dist = {
    'n_estimators': sp_randint(50, 300),
    'max_depth': sp_randint(3, 15),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_split': sp_randint(2, 11),
    'min_samples_leaf': sp_randint(1, 5),
    'bootstrap': [True, False]
}

# Search settings
n_iter = 20
cv = 3
scoring = 'f1'
random_state = 74

# Random Forest tuning
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=random_state),
    rf_param_dist,
    n_iter=n_iter,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    random_state=random_state,
    verbose=1
)

print("Tuning Random Forest hyperparameters...")
rf_search.fit(X_train_scaled, y_train_balanced)
print("\nBest score (cv):", rf_search.best_score_)
print("Best parameters:", rf_search.best_params_)

# Train final model with best parameters
rf_best = RandomForestClassifier(**rf_search.best_params_, random_state=74)
rf_best.fit(X_train_scaled, y_train_balanced)

# Evaluate final model
rf_best_results = evaluate_model(rf_best, X_train_scaled, X_test_scaled, y_train_balanced, y_test)
results['Random Forest (tuned)'] = rf_best_results
models['Random Forest (tuned)'] = rf_best

# Update results DataFrame
results_df = pd.DataFrame(results).round(4)
print("\nUpdated Model Comparison:")
display(results_df)


# Get best model based on F1 score
best_model_name = results_df.loc['f1'].idxmax()
best_model = models[best_model_name]

# Get predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.show()

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=importances.head(15))
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    display(importances.head(10))


# Balance data using SMOTE (only for training)
smote = SMOTE(random_state=74, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Fit the scaler on the balanced data
scaler = RobustScaler().fit(X_res)

# Train Random Forest on the scaled data
X_res_scaled = scaler.transform(X_res)
rf = RandomForestClassifier(
    n_estimators=295,
    max_depth=14,
    max_features='auto',
    min_samples_split=2,
    min_samples_leaf=2,
    bootstrap=False,
    random_state=74
)
rf.fit(X_res_scaled, y_res)

# --- Build Inference Pipeline ---
inference_pipeline = Pipeline([
    ('scaler', scaler),
    ('model', rf)
])

# --- Define the Predictor Class ---
class BankMarketingPredictor:
    def __init__(self, model_pipeline, feature_names):
        self.model = model_pipeline
        self.feature_names = feature_names

    def preprocess_data(self, data, is_single_sample=False):
        """Preprocess input to match training format."""
        if is_single_sample:
            data = pd.DataFrame([data])

        df = data.copy()

        # Drop unnecessary columns if exist
        drop_cols = ['SampleId', 'default', 'loan']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Age binning
        df['age'] = pd.cut(df['age'],
                           bins=[17, 29, 45, 60, 95],
                           labels=['young', 'adult', 'middle_aged', 'senior'])

        # Log transform numeric features
        for col in ['balance', 'duration', 'campaign']:
            min_val = df[col].min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            df[col] = np.log1p(df[col] + shift)

        # Month encoding
        month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                     'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        df['month_num'] = df['month'].map(month_map)
        df['month_sin'] = np.sin(2*np.pi*df['month_num']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month_num']/12)
        df.drop(columns=['month', 'month_num'], inplace=True)

        # One-hot encoding for categoricals
        df = pd.get_dummies(df,
                            columns=['job','marital','contact','poutcome','age'],
                            drop_first=True)

        # Ordinal encoding
        df['education'] = df['education'].map({'unknown':0,'primary':1,'secondary':2,'tertiary':3})

        # Binary encoding
        df['housing'] = df['housing'].map({'yes':1,'no':0})

        # Add missing columns (if needed)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_names]

    def predict(self, data):
        """Make predictions on single or batch data."""
        is_single = isinstance(data, dict)
        pre_data = self.preprocess_data(data, is_single_sample=is_single)

        preds = self.model.predict(pre_data)
        sample_ids = pd.Series(range(len(preds)))
        results = pd.DataFrame({'SampleId': sample_ids, 'y': preds.astype(int)})
        return results


# --- Initialize Predictor ---
predictor = BankMarketingPredictor(inference_pipeline, X.columns)

# --- Example: Single Sample Prediction ---
sample_data = {
    'age': 41,
    'job': 'entrepreneur',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'balance': 1500,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'duration': 240,
    'campaign': 2,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown'
}

# Make prediction on single sample
print("Single Sample Prediction:")
print("-" * 50)
single_prediction = predictor.predict(sample_data)
print(single_prediction)

print("\nPrediction explanation:")
print("0 = No subscription")
print("1 = Will subscribe")


# Example 2: Batch Predictions
# Load test data
test_data = pd.read_csv("/kaggle/input/bank-marketing-prediction/test.csv")

# Make predictions on test data
print("Batch Predictions:")
print("-" * 50)
batch_predictions = predictor.predict(test_data)

# Display summary of predictions
positives = batch_predictions['y'].sum()
total = len(batch_predictions)
print(f"\nPrediction Summary:")
print(f"Total predictions: {total}")
print(f"Positive predictions (y=1): {positives:,d} ({(positives/total)*100:.1f}%)")
print(f"Negative predictions (y=0): {total-positives:,d} ({((total-positives)/total)*100:.1f}%)")

# Save predictions to file
output_path = "../data/submission/notebook_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
batch_predictions.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

# Display first few predictions
print("\nFirst 10 predictions:")
print(batch_predictions.head(10))

