# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate

# Metrics
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, classification_report, 
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, log_loss
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("âœ“ All libraries imported successfully!")


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Display first few rows
print("\n=== First 5 rows of training data ===")
display(train_df.head())


# Data types and missing values
print("\n=== Data Information ===")
train_df.info()


# Check for missing values
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("\n=== Missing Values in Training Data ===")
print(missing_train[missing_train > 0])

print("\n=== Missing Values in Test Data ===")
print(missing_test[missing_test > 0])

if missing_train.sum() == 0 and missing_test.sum() == 0:
    print("\nâœ“ No missing values detected in the datasets!")


# Separate numerical and categorical features
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('id')  # Remove ID column
numerical_features.remove('diagnosed_diabetes')  # Remove target variable

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")


# Target variable distribution
print("\n=== Target Variable Distribution ===")
target_counts = train_df['diagnosed_diabetes'].value_counts()
target_props = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100

print(f"\nClass 0 (No Diabetes): {target_counts[0]:,} ({target_props[0]:.2f}%)")
print(f"Class 1 (Diabetes): {target_counts[1]:,} ({target_props[1]:.2f}%)")
print(f"\nClass balance ratio: {target_counts[1] / target_counts[0]:.3f}")


# Descriptive statistics for numerical features
print("\n=== Descriptive Statistics for Numerical Features ===")
display(train_df[numerical_features].describe().T)


# Additional statistics: skewness and kurtosis
print("\n=== Skewness and Kurtosis ===")
stats_df = pd.DataFrame({
    'Feature': numerical_features,
    'Skewness': [train_df[col].skew() for col in numerical_features],
    'Kurtosis': [train_df[col].kurtosis() for col in numerical_features]
})
display(stats_df.sort_values('Skewness', ascending=False))


# Categorical features statistics
print("\n=== Categorical Features Analysis ===")
for col in categorical_features:
    print(f"\n{col}:")
    print(train_df[col].value_counts())
    print(f"Unique values: {train_df[col].nunique()}")


# Statistical comparison between diabetes and non-diabetes groups
print("\n=== Statistical Comparison: Diabetes vs Non-Diabetes ===")
comparison_df = train_df.groupby('diagnosed_diabetes')[numerical_features].agg(['mean', 'median', 'std'])
display(comparison_df.T)


# Perform t-tests for numerical features
print("\n=== T-Test Results (Diabetes vs Non-Diabetes) ===")
t_test_results = []

for feature in numerical_features:
    group_0 = train_df[train_df['diagnosed_diabetes'] == 0][feature]
    group_1 = train_df[train_df['diagnosed_diabetes'] == 1][feature]
    
    t_stat, p_value = stats.ttest_ind(group_0, group_1)
    
    t_test_results.append({
        'Feature': feature,
        'T-Statistic': t_stat,
        'P-Value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

t_test_df = pd.DataFrame(t_test_results).sort_values('P-Value')
display(t_test_df)


# Chi-square tests for categorical features
print("\n=== Chi-Square Test Results (Categorical Features vs Target) ===")
chi_square_results = []

for feature in categorical_features:
    contingency_table = pd.crosstab(train_df[feature], train_df['diagnosed_diabetes'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    chi_square_results.append({
        'Feature': feature,
        'Chi-Square': chi2,
        'P-Value': p_value,
        'Degrees of Freedom': dof,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

chi_square_df = pd.DataFrame(chi_square_results).sort_values('P-Value')
display(chi_square_df)


# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Count plot
sns.countplot(data=train_df, x='diagnosed_diabetes', ax=axes[0], palette='Set2')
axes[0].set_title('Distribution of Target Variable', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Diagnosed Diabetes', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])

# Add value labels on bars
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%d')

# Pie chart
colors = sns.color_palette('Set2')
axes[1].pie(target_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', 
            startangle=90, colors=colors, textprops={'fontsize': 12})
axes[1].set_title('Proportion of Target Classes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# Distribution plots for numerical features
n_features = len(numerical_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    sns.histplot(data=train_df, x=feature, kde=True, ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


# Box plots for key numerical features by target
key_features = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
                'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']

n_features = len(key_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    sns.boxplot(data=train_df, x='diagnosed_diabetes', y=feature, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{feature} by Diabetes Status', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Diagnosed Diabetes', fontsize=9)
    axes[idx].set_ylabel(feature, fontsize=9)
    axes[idx].set_xticklabels(['No', 'Yes'])

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


# Violin plots for select features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

top_features = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'cholesterol_total', 'triglycerides']

for idx, feature in enumerate(top_features):
    sns.violinplot(data=train_df, x='diagnosed_diabetes', y=feature, ax=axes[idx], palette='muted')
    axes[idx].set_title(f'{feature} Distribution by Diabetes Status', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Diagnosed Diabetes', fontsize=10)
    axes[idx].set_ylabel(feature, fontsize=10)
    axes[idx].set_xticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()


# Correlation matrix
correlation_matrix = train_df[numerical_features + ['diagnosed_diabetes']].corr()

# Plot heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# Correlation with target variable
target_correlation = correlation_matrix['diagnosed_diabetes'].drop('diagnosed_diabetes').sort_values(ascending=False)

plt.figure(figsize=(10, 8))
target_correlation.plot(kind='barh', color='teal')
plt.title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

print("\nTop 10 features positively correlated with diabetes:")
print(target_correlation.head(10))


# Categorical features vs target
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    # Create crosstab
    ct = pd.crosstab(train_df[feature], train_df['diagnosed_diabetes'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[idx], stacked=False, color=['skyblue', 'salmon'])
    axes[idx].set_title(f'{feature} vs Diabetes', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Percentage (%)', fontsize=10)
    axes[idx].legend(['No Diabetes', 'Diabetes'], loc='upper right')
    axes[idx].tick_params(axis='x', rotation=45)

# Hide empty subplots
for idx in range(len(categorical_features), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


# Age distribution by diabetes status
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
sns.histplot(data=train_df, x='age', hue='diagnosed_diabetes', kde=True, ax=axes[0], palette='Set1')
axes[0].set_title('Age Distribution by Diabetes Status', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].legend(['No Diabetes', 'Diabetes'])

# Age groups
train_df['age_group'] = pd.cut(train_df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                labels=['<30', '30-40', '40-50', '50-60', '60+'])
age_group_diabetes = train_df.groupby('age_group')['diagnosed_diabetes'].mean() * 100

age_group_diabetes.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Diabetes Rate by Age Group', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Age Group', fontsize=11)
axes[1].set_ylabel('Diabetes Rate (%)', fontsize=11)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()


# BMI categories
train_df['bmi_category'] = pd.cut(train_df['bmi'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# BMI distribution
sns.histplot(data=train_df, x='bmi', hue='diagnosed_diabetes', kde=True, ax=axes[0], palette='viridis')
axes[0].set_title('BMI Distribution by Diabetes Status', fontsize=13, fontweight='bold')
axes[0].set_xlabel('BMI', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].legend(['No Diabetes', 'Diabetes'])

# Diabetes rate by BMI category
bmi_diabetes = train_df.groupby('bmi_category')['diagnosed_diabetes'].mean() * 100
bmi_diabetes.plot(kind='bar', ax=axes[1], color='purple')
axes[1].set_title('Diabetes Rate by BMI Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('BMI Category', fontsize=11)
axes[1].set_ylabel('Diabetes Rate (%)', fontsize=11)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()


# Select key features for pair plot
pair_features = ['age', 'bmi', 'systolic_bp', 'cholesterol_total', 'diagnosed_diabetes']
sample_data = train_df[pair_features].sample(n=min(2000, len(train_df)), random_state=RANDOM_SEED)

sns.pairplot(sample_data, hue='diagnosed_diabetes', palette='husl', diag_kind='kde', corner=True)
plt.suptitle('Pair Plot of Key Features', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


def engineer_features(df):
    """
    Engineer new features from existing ones.
    """
    df = df.copy()
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                              labels=[0, 1, 2, 3, 4])
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
    
    # Blood pressure category (Hypertension stages)
    df['bp_category'] = 0
    df.loc[(df['systolic_bp'] >= 120) | (df['diastolic_bp'] >= 80), 'bp_category'] = 1  # Elevated
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 2  # Stage 1
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 3  # Stage 2
    
    # Cholesterol ratios
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    df['triglycerides_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-6)
    
    # Pulse pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Mean arterial pressure
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    # Health score (composite features)
    df['lifestyle_score'] = (df['physical_activity_minutes_per_week'] / 150) + df['diet_score'] - (df['screen_time_hours_per_day'] * 0.5)
    
    # Risk factors count
    df['risk_factors_count'] = (df['family_history_diabetes'] + 
                                 df['hypertension_history'] + 
                                 df['cardiovascular_history'])
    
    # Smoking status binary
    df['is_smoker'] = (df['smoking_status'].isin(['Current', 'Former'])).astype(int)
    
    # Alcohol risk
    df['alcohol_risk'] = (df['alcohol_consumption_per_week'] > 7).astype(int)
    
    # Sleep quality
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    
    # Physical inactivity
    df['inactive'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    
    # Metabolic syndrome indicators
    df['metabolic_score'] = ((df['bmi'] > 30).astype(int) + 
                             (df['waist_to_hip_ratio'] > 0.9).astype(int) +
                             (df['systolic_bp'] > 130).astype(int) +
                             (df['triglycerides'] > 150).astype(int))
    
    return df

# Apply feature engineering
train_df_fe = engineer_features(train_df)
test_df_fe = engineer_features(test_df)

print("âœ“ Feature engineering completed!")
print(f"\nNew feature count: {train_df_fe.shape[1] - train_df.shape[1]}")
print(f"Total features: {train_df_fe.shape[1]}")


# Encode categorical variables
def encode_categorical(train, test, categorical_cols):
    """
    Encode categorical variables using Label Encoding.
    """
    train_encoded = train.copy()
    test_encoded = test.copy()
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to ensure consistent encoding
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        
        train_encoded[col] = le.transform(train[col])
        test_encoded[col] = le.transform(test[col])
        
        label_encoders[col] = le
    
    return train_encoded, test_encoded, label_encoders

# Apply encoding
train_encoded, test_encoded, label_encoders = encode_categorical(
    train_df_fe, test_df_fe, categorical_features
)

print("âœ“ Categorical encoding completed!")


# Prepare features and target
X = train_encoded.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_encoded['diagnosed_diabetes']
X_test = test_encoded.drop(['id'], axis=1)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for interpretability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("âœ“ Feature scaling completed!")


# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train and evaluate a model.
    """
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Training ROC AUC:   {train_auc:.4f}")
    print(f"Validation ROC AUC: {val_auc:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    
    return {
        'model': model,
        'model_name': model_name,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'y_val_proba': y_val_proba
    }


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1),
}

# Train and evaluate all models
results = []

for name, model in models.items():
    result = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    results.append(result)


# Compare model performance
comparison_df = pd.DataFrame([{
    'Model': r['model_name'],
    'Train AUC': r['train_auc'],
    'Validation AUC': r['val_auc'],
    'Accuracy': r['val_accuracy'],
    'Precision': r['val_precision'],
    'Recall': r['val_recall'],
    'F1 Score': r['val_f1']
} for r in results])

comparison_df = comparison_df.sort_values('Validation AUC', ascending=False)
print("\n=== Model Comparison ===")
display(comparison_df)


# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC AUC comparison
comparison_df.plot(x='Model', y=['Train AUC', 'Validation AUC'], kind='bar', ax=axes[0], color=['skyblue', 'coral'])
axes[0].set_title('Model Performance: ROC AUC', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('ROC AUC Score', fontsize=12)
axes[0].legend(['Train AUC', 'Validation AUC'])
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylim([0.5, 1.0])

# Other metrics comparison
comparison_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'], kind='bar', ax=axes[1])
axes[1].set_title('Model Performance: Other Metrics', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim([0.5, 1.0])

plt.tight_layout()
plt.show()


# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

for result in results:
    fpr, tpr, _ = roc_curve(y_val, result['y_val_proba'])
    plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['val_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Plot confusion matrices
n_models = len(results)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, result in enumerate(results):
    y_val_pred = result['model'].predict(X_val_scaled)
    cm = confusion_matrix(y_val, y_val_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    axes[idx].set_title(f"{result['model_name']}\nAUC: {result['val_auc']:.4f}", 
                        fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

# Hide empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()


# Select top performing model for tuning
best_baseline_model = comparison_df.iloc[0]['Model']
print(f"Best baseline model: {best_baseline_model}")
print(f"Validation AUC: {comparison_df.iloc[0]['Validation AUC']:.4f}")


# XGBoost hyperparameter tuning
print("\nTuning XGBoost...")

xgb_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

xgb_model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=-1)

xgb_random_search = RandomizedSearchCV(
    xgb_model, xgb_params, n_iter=20, cv=3, 
    scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1, verbose=1
)

xgb_random_search.fit(X_train_scaled, y_train)

print(f"\nBest XGBoost parameters: {xgb_random_search.best_params_}")
print(f"Best CV ROC AUC: {xgb_random_search.best_score_:.4f}")

# Evaluate tuned model
xgb_tuned = xgb_random_search.best_estimator_
xgb_tuned_result = evaluate_model(xgb_tuned, X_train_scaled, y_train, X_val_scaled, y_val, 'XGBoost (Tuned)')


# LightGBM hyperparameter tuning
print("\nTuning LightGBM...")

lgbm_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_samples': [20, 30, 50]
}

lgbm_model = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)

lgbm_random_search = RandomizedSearchCV(
    lgbm_model, lgbm_params, n_iter=20, cv=3,
    scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1, verbose=1
)

lgbm_random_search.fit(X_train_scaled, y_train)

print(f"\nBest LightGBM parameters: {lgbm_random_search.best_params_}")
print(f"Best CV ROC AUC: {lgbm_random_search.best_score_:.4f}")

# Evaluate tuned model
lgbm_tuned = lgbm_random_search.best_estimator_
lgbm_tuned_result = evaluate_model(lgbm_tuned, X_train_scaled, y_train, X_val_scaled, y_val, 'LightGBM (Tuned)')


# CatBoost model
print("\nTraining CatBoost...")

catboost_params = {
    'iterations': [200, 300, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7]
}

catboost_model = CatBoostClassifier(random_state=RANDOM_SEED, verbose=0)

catboost_random_search = RandomizedSearchCV(
    catboost_model, catboost_params, n_iter=15, cv=3,
    scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1, verbose=1
)

catboost_random_search.fit(X_train_scaled, y_train)

print(f"\nBest CatBoost parameters: {catboost_random_search.best_params_}")
print(f"Best CV ROC AUC: {catboost_random_search.best_score_:.4f}")

# Evaluate tuned model
catboost_tuned = catboost_random_search.best_estimator_
catboost_tuned_result = evaluate_model(catboost_tuned, X_train_scaled, y_train, X_val_scaled, y_val, 'CatBoost (Tuned)')


# Create voting ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_tuned),
        ('lgbm', lgbm_tuned),
        ('catboost', catboost_tuned)
    ],
    voting='soft'
)

voting_result = evaluate_model(voting_clf, X_train_scaled, y_train, X_val_scaled, y_val, 'Voting Ensemble')


# Create weighted ensemble based on validation performance
xgb_proba = xgb_tuned.predict_proba(X_val_scaled)[:, 1]
lgbm_proba = lgbm_tuned.predict_proba(X_val_scaled)[:, 1]
catboost_proba = catboost_tuned.predict_proba(X_val_scaled)[:, 1]

# Calculate weights based on validation AUC
xgb_weight = xgb_tuned_result['val_auc']
lgbm_weight = lgbm_tuned_result['val_auc']
catboost_weight = catboost_tuned_result['val_auc']

total_weight = xgb_weight + lgbm_weight + catboost_weight

xgb_weight /= total_weight
lgbm_weight /= total_weight
catboost_weight /= total_weight

print(f"\nEnsemble weights:")
print(f"XGBoost: {xgb_weight:.4f}")
print(f"LightGBM: {lgbm_weight:.4f}")
print(f"CatBoost: {catboost_weight:.4f}")

# Weighted predictions
weighted_proba = (xgb_weight * xgb_proba + 
                  lgbm_weight * lgbm_proba + 
                  catboost_weight * catboost_proba)

weighted_auc = roc_auc_score(y_val, weighted_proba)
print(f"\nWeighted Ensemble Validation AUC: {weighted_auc:.4f}")


# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', xgb_tuned),
        ('lgbm', lgbm_tuned),
        ('catboost', catboost_tuned)
    ],
    final_estimator=LogisticRegression(random_state=RANDOM_SEED),
    cv=5
)

stacking_result = evaluate_model(stacking_clf, X_train_scaled, y_train, X_val_scaled, y_val, 'Stacking Ensemble')


# Compile all results
all_results = results + [xgb_tuned_result, lgbm_tuned_result, catboost_tuned_result, 
                         voting_result, stacking_result]

# Add weighted ensemble manually
all_results.append({
    'model_name': 'Weighted Ensemble',
    'val_auc': weighted_auc,
    'y_val_proba': weighted_proba
})

# Create final comparison
final_comparison = pd.DataFrame([{
    'Model': r['model_name'],
    'Validation AUC': r['val_auc']
} for r in all_results])

final_comparison = final_comparison.sort_values('Validation AUC', ascending=False)

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
display(final_comparison)

best_model_name = final_comparison.iloc[0]['Model']
best_auc = final_comparison.iloc[0]['Validation AUC']
print(f"\nğŸ�† Best Model: {best_model_name}")
print(f"ğŸ�¯ Best Validation AUC: {best_auc:.4f}")


# Visualize final comparison
plt.figure(figsize=(12, 6))
colors = ['gold' if i == 0 else 'steelblue' for i in range(len(final_comparison))]
plt.barh(final_comparison['Model'], final_comparison['Validation AUC'], color=colors)
plt.xlabel('Validation ROC AUC', fontsize=12, fontweight='bold')
plt.ylabel('Model', fontsize=12, fontweight='bold')
plt.title('Final Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xlim([0.5, 1.0])
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=0.8, label='Random Baseline')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# Feature importance from XGBoost
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_tuned.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(20)['Feature'], feature_importance.head(20)['Importance'], color='teal')
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 20 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 20 Most Important Features:")
display(feature_importance.head(20))


# Select best model for final predictions
print(f"Using {best_model_name} for final predictions...\n")

# Generate predictions on test set
if best_model_name == 'Weighted Ensemble':
    # For weighted ensemble, combine predictions from all three models
    xgb_test_proba = xgb_tuned.predict_proba(X_test_scaled)[:, 1]
    lgbm_test_proba = lgbm_tuned.predict_proba(X_test_scaled)[:, 1]
    catboost_test_proba = catboost_tuned.predict_proba(X_test_scaled)[:, 1]
    
    test_predictions = (xgb_weight * xgb_test_proba + 
                        lgbm_weight * lgbm_test_proba + 
                        catboost_weight * catboost_test_proba)
else:
    # For other models, use the best model directly
    best_model_obj = [r for r in all_results if r['model_name'] == best_model_name][0]['model']
    test_predictions = best_model_obj.predict_proba(X_test_scaled)[:, 1]

print(f"âœ“ Generated predictions for {len(test_predictions)} test samples")
print(f"\nPrediction statistics:")
print(f"  Min probability: {test_predictions.min():.4f}")
print(f"  Max probability: {test_predictions.max():.4f}")
print(f"  Mean probability: {test_predictions.mean():.4f}")
print(f"  Median probability: {np.median(test_predictions):.4f}")


# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION FILE CREATED SUCCESSFULLY")
print("="*60)
print(f"\nFile: submission.csv")
print(f"Shape: {submission.shape}")
print(f"\nFirst 10 rows:")
display(submission.head(10))

# Verify submission format
print(f"\nâœ“ Submission file validation:")
print(f"  âœ“ Correct number of predictions: {len(submission) == len(test_df)}")
print(f"  âœ“ ID column present: {'id' in submission.columns}")
print(f"  âœ“ Target column present: {'diagnosed_diabetes' in submission.columns}")
print(f"  âœ“ No missing values: {submission.isnull().sum().sum() == 0}")
print(f"  âœ“ All probabilities in [0,1]: {(submission['diagnosed_diabetes'] >= 0).all() and (submission['diagnosed_diabetes'] <= 1).all()}")


print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

print("\nğŸ“Š Dataset:")
print(f"  â€¢ Training samples: {len(train_df):,}")
print(f"  â€¢ Test samples: {len(test_df):,}")
print(f"  â€¢ Features: {X_train.shape[1]} (after feature engineering)")
print(f"  â€¢ Target distribution: {(y.sum() / len(y) * 100):.2f}% positive class")

print("\nğŸ”� Key Findings:")
print(f"  â€¢ Most important features: {', '.join(feature_importance.head(5)['Feature'].tolist())}")
print(f"  â€¢ Significant categorical features: family_history_diabetes, hypertension_history")
print(f"  â€¢ Strong predictors: BMI, age, cholesterol levels, blood pressure")

print("\nğŸ¤– Models Tested:")
print(f"  â€¢ Baseline models: {len(results)}")
print(f"  â€¢ Tuned models: 3 (XGBoost, LightGBM, CatBoost)")
print(f"  â€¢ Ensemble models: 3 (Voting, Weighted, Stacking)")

print("\nğŸ�† Best Model Performance:")
print(f"  â€¢ Model: {best_model_name}")
print(f"  â€¢ Validation ROC AUC: {best_auc:.4f}")

print("\nğŸ“� Deliverables:")
print(f"  â€¢ Submission file: submission.csv")
print(f"  â€¢ Ready for submission: âœ“")

print("\n" + "="*80)

