print("hello")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

print("Dataset Shapes:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Sample Submission: {sample_submission.shape}")



!pip install plotly


def analyze_dataset_structure(train_df, test_df, sample_submission):
    """Comprehensive analysis of dataset structure"""
    
    print("="*50)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*50)
    
    # Basic info
    print("\nğŸ“Š BASIC INFORMATION:")
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Total features in train: {train_df.shape[1]}")
    print(f"Total features in test: {test_df.shape[1]}")
    
    # Column analysis
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    print(f"\nğŸ”� COLUMN ANALYSIS:")
    print(f"Common columns: {len(train_cols & test_cols)}")
    print(f"Train-only columns: {list(train_cols - test_cols)}")
    print(f"Test-only columns: {list(test_cols - train_cols)}")
    
    # Target variable identification
    target_col = list(train_cols - test_cols)[0] if (train_cols - test_cols) else None
    print(f"Target variable: {target_col}")
    
    # Data types
    print(f"\nğŸ“‹ DATA TYPES:")
    print("Training set:")
    print(train_df.dtypes.value_counts())
    
    return target_col

target_column = analyze_dataset_structure(train_df, test_df, sample_submission)



def comprehensive_missing_analysis(train_df, test_df):
    """Detailed missing value analysis"""
    
    print("="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Missing values in train
    train_missing = train_df.isnull().sum()
    train_missing_pct = (train_missing / len(train_df)) * 100
    
    # Missing values in test
    test_missing = test_df.isnull().sum()
    test_missing_pct = (test_missing / len(test_df)) * 100
    
    # Create missing values dataframe
    missing_df = pd.DataFrame({
        'Feature': train_df.columns,
        'Train_Missing': train_missing.values,
        'Train_Missing_Pct': train_missing_pct.values,
        'Test_Missing': test_missing.reindex(train_df.columns, fill_value=0).values,
        'Test_Missing_Pct': test_missing_pct.reindex(train_df.columns, fill_value=0).values
    })
    
    missing_df = missing_df[(missing_df['Train_Missing'] > 0) | (missing_df['Test_Missing'] > 0)]
    missing_df = missing_df.sort_values('Train_Missing_Pct', ascending=False)
    
    if len(missing_df) > 0:
        print("Features with missing values:")
        print(missing_df.to_string(index=False))
        
        # Visualize missing values
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Missing values heatmap for train
        if train_missing.sum() > 0:
            missing_cols = train_missing[train_missing > 0].index
            sns.heatmap(train_df[missing_cols].isnull(), cbar=True, ax=axes[0])
            axes[0].set_title('Missing Values Pattern - Training Set')
        
        # Missing values comparison
        if len(missing_df) > 0:
            missing_df.set_index('Feature')[['Train_Missing_Pct', 'Test_Missing_Pct']].plot(
                kind='bar', ax=axes[1]
            )
            axes[1].set_title('Missing Values Percentage Comparison')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        print("âœ… No missing values found in the dataset!")
    
    return missing_df

missing_analysis = comprehensive_missing_analysis(train_df, test_df)



def analyze_target_variable(train_df, target_col):
    """Comprehensive target variable analysis"""
    
    if target_col is None:
        print("No target variable identified!")
        return None
    
    print("="*50)
    print(f"TARGET VARIABLE ANALYSIS: {target_col}")
    print("="*50)
    
    target_data = train_df[target_col]
    
    # Basic statistics
    print(f"\nğŸ“Š BASIC STATISTICS:")
    print(f"Data type: {target_data.dtype}")
    print(f"Unique values: {target_data.nunique()}")
    print(f"Missing values: {target_data.isnull().sum()}")
    
    # Determine if classification or regression
    is_classification = target_data.nunique() < 20 or target_data.dtype == 'object'
    
    if is_classification:
        print(f"Task Type: CLASSIFICATION")
        print(f"\nğŸ�¯ CLASS DISTRIBUTION:")
        class_counts = target_data.value_counts()
        class_props = target_data.value_counts(normalize=True)
        
        for class_val, count in class_counts.items():
            print(f"Class {class_val}: {count:,} ({class_props[class_val]:.2%})")
        
        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Target Class Distribution')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Target Class Proportions')
        
        plt.tight_layout()
        plt.show()
        
        # Check for class imbalance
        min_class_prop = class_props.min()
        max_class_prop = class_props.max()
        imbalance_ratio = max_class_prop / min_class_prop
        
        if imbalance_ratio > 2:
            print(f"âš ï¸� CLASS IMBALANCE DETECTED: Ratio = {imbalance_ratio:.2f}")
        else:
            print("âœ… Classes are relatively balanced")
            
    else:
        print(f"Task Type: REGRESSION")
        print(f"\nğŸ“ˆ DISTRIBUTION STATISTICS:")
        print(target_data.describe())
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        axes[0,0].hist(target_data, bins=50, alpha=0.7)
        axes[0,0].set_title('Target Distribution')
        axes[0,0].set_xlabel(target_col)
        
        # Box plot
        axes[0,1].boxplot(target_data)
        axes[0,1].set_title('Target Box Plot')
        axes[0,1].set_ylabel(target_col)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target_data, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Log transformation (if positive values)
        if target_data.min() > 0:
            axes[1,1].hist(np.log(target_data), bins=50, alpha=0.7)
            axes[1,1].set_title('Log-Transformed Target Distribution')
        
        plt.tight_layout()
        plt.show()
    
    return {'type': 'classification' if is_classification else 'regression', 'stats': target_data.describe()}

target_analysis = analyze_target_variable(train_df, target_column)



def comprehensive_feature_analysis(train_df, target_col):
    """Analyze all features comprehensively"""
    
    print("="*50)
    print("COMPREHENSIVE FEATURE ANALYSIS")
    print("="*50)
    
    # Separate features by type
    features = [col for col in train_df.columns if col != target_col]
    
    numeric_features = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_df[features].select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nğŸ“Š FEATURE BREAKDOWN:")
    print(f"Total features: {len(features)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Analyze numeric features
    if numeric_features:
        print(f"\nğŸ”¢ NUMERIC FEATURES ANALYSIS:")
        numeric_stats = train_df[numeric_features].describe()
        print(numeric_stats)
        
        # Check for potential issues
        print(f"\nğŸš¨ POTENTIAL ISSUES:")
        
        # Zero variance features
        zero_var_features = []
        for col in numeric_features:
            if train_df[col].std() == 0:
                zero_var_features.append(col)
        
        if zero_var_features:
            print(f"Zero variance features: {zero_var_features}")
        
        # High cardinality numeric features (might be IDs)
        high_cardinality = []
        for col in numeric_features:
            unique_ratio = train_df[col].nunique() / len(train_df)
            if unique_ratio > 0.95:
                high_cardinality.append((col, unique_ratio))
        
        if high_cardinality:
            print("High cardinality numeric features (possible IDs):")
            for col, ratio in high_cardinality:
                print(f"  {col}: {ratio:.3f} unique ratio")
    
    # Analyze categorical features
    if categorical_features:
        print(f"\nğŸ“� CATEGORICAL FEATURES ANALYSIS:")
        for col in categorical_features:
            unique_count = train_df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            
            if unique_count <= 10:
                print(f"  Values: {train_df[col].value_counts().to_dict()}")
            else:
                top_values = train_df[col].value_counts().head()
                print(f"  Top values: {top_values.to_dict()}")
    
    return {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'total_features': len(features)
    }

feature_analysis = comprehensive_feature_analysis(train_df, target_column)


def distribution_and_outlier_analysis(train_df, numeric_features, target_col):
    """Comprehensive distribution and outlier analysis"""
    
    print("="*50)
    print("DISTRIBUTION AND OUTLIER ANALYSIS")
    print("="*50)
    
    if len(numeric_features) > 0:
        # Select top features for detailed analysis
        analysis_features = numeric_features[:12]  # Limit to avoid overcrowding
        
        # Distribution plots
        n_cols = 3
        n_rows = (len(analysis_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        outlier_summary = {}
        
        for i, feature in enumerate(analysis_features):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(train_df[feature], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution: {feature}')
                
                # Outlier detection using IQR
                Q1 = train_df[feature].quantile(0.25)
                Q3 = train_df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = train_df[(train_df[feature] < lower_bound) | 
                                  (train_df[feature] > upper_bound)]
                outlier_summary[feature] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(train_df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Remove empty subplots
        for i in range(len(analysis_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Outlier summary
        print(f"\nğŸš¨ OUTLIER SUMMARY:")
        outlier_df = pd.DataFrame(outlier_summary).T
        outlier_df = outlier_df.sort_values('percentage', ascending=False)
        print(outlier_df)
        
        # Box plots for features with significant outliers
        high_outlier_features = outlier_df[outlier_df['percentage'] > 5].index.tolist()
        
        if high_outlier_features:
            n_features = len(high_outlier_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, feature in enumerate(high_outlier_features):
                if i < len(axes):
                    sns.boxplot(y=train_df[feature], ax=axes[i])
                    axes[i].set_title(f'Box Plot: {feature}')
            
            plt.tight_layout()
            plt.show()

distribution_and_outlier_analysis(train_df, feature_analysis['numeric_features'], target_column)



def feature_importance_analysis(train_df, target_col, numeric_features, categorical_features):
    """Feature importance analysis using various methods"""
    
    print("="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    
    # Prepare data
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Handle categorical variables
    X_processed = X.copy()
    label_encoders = {}
    
    for col in categorical_features:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
    
    # Determine task type
    is_classification = y.nunique() < 20 or y.dtype == 'object'
    
    if is_classification and y.dtype == 'object':
        le_target = LabelEncoder()
        y_processed = le_target.fit_transform(y)
    else:
        y_processed = y
    
    # Random Forest Feature Importance
    if is_classification:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        mutual_info_func = mutual_info_classif
    else:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        mutual_info_func = mutual_info_regression
    
    rf_model.fit(X_processed, y_processed)
    rf_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'rf_importance': rf_model.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Mutual Information
    mi_scores = mutual_info_func(X_processed, y_processed, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    # Combine importance scores
    importance_combined = rf_importance.merge(mi_importance, on='feature')
    importance_combined['combined_score'] = (
        importance_combined['rf_importance'] + importance_combined['mutual_info']
    ) / 2
    importance_combined = importance_combined.sort_values('combined_score', ascending=False)
    
    print("ğŸ�† TOP 15 MOST IMPORTANT FEATURES:")
    print(importance_combined.head(15).to_string(index=False))
    
    # Visualize feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Random Forest Importance
    top_rf = rf_importance.head(15)
    axes[0].barh(range(len(top_rf)), top_rf['rf_importance'])
    axes[0].set_yticks(range(len(top_rf)))
    axes[0].set_yticklabels(top_rf['feature'])
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].invert_yaxis()
    
    # Mutual Information
    top_mi = mi_importance.head(15)
    axes[1].barh(range(len(top_mi)), top_mi['mutual_info'])
    axes[1].set_yticks(range(len(top_mi)))
    axes[1].set_yticklabels(top_mi['feature'])
    axes[1].set_title('Mutual Information Scores')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return importance_combined

feature_importance = feature_importance_analysis(
    train_df, target_column, 
    feature_analysis['numeric_features'], 
    feature_analysis['categorical_features']
)



def generate_eda_summary_and_recommendations(train_df, test_df, target_analysis, feature_analysis, missing_analysis):
    """Generate comprehensive EDA summary and ML recommendations"""
    
    print("="*60)
    print("ğŸ�¯ EDA SUMMARY AND ML RECOMMENDATIONS")
    print("="*60)
    
    print(f"\nğŸ“Š DATASET OVERVIEW:")
    print(f"â€¢ Training samples: {len(train_df):,}")
    print(f"â€¢ Test samples: {len(test_df):,}")
    print(f"â€¢ Total features: {feature_analysis['total_features']}")
    print(f"â€¢ Numeric features: {len(feature_analysis['numeric_features'])}")
    print(f"â€¢ Categorical features: {len(feature_analysis['categorical_features'])}")
    
    print(f"\nğŸ�¯ PROBLEM TYPE:")
    if target_analysis:
        print(f"â€¢ Task: {target_analysis['type'].upper()}")
    
    print(f"\nğŸ”§ PREPROCESSING RECOMMENDATIONS:")
    
    # Missing values
    if len(missing_analysis) > 0:
        print("â€¢ Handle missing values:")
        for _, row in missing_analysis.head().iterrows():
            if row['Train_Missing_Pct'] > 0:
                print(f"  - {row['Feature']}: {row['Train_Missing_Pct']:.1f}% missing")
    else:
        print("â€¢ âœ… No missing values to handle")
    
    # Feature engineering suggestions
    print("â€¢ Feature Engineering Opportunities:")
    print("  - Create interaction features between top important features")
    print("  - Consider polynomial features for non-linear relationships")
    print("  - Apply feature scaling/normalization for numeric features")
    
    if len(feature_analysis['categorical_features']) > 0:
        print("  - Encode categorical variables (Label/One-hot encoding)")
    
    print(f"\nğŸ¤– MODEL RECOMMENDATIONS:")
    print("â€¢ Start with these models:")
    print("  - Random Forest (baseline)")
    print("  - XGBoost/LightGBM (likely best performance)")
    print("  - Neural Networks (if sufficient data)")
    
    if target_analysis and target_analysis['type'] == 'classification':
        print("â€¢ Classification-specific:")
        print("  - Use stratified cross-validation")
        print("  - Consider ensemble methods")
        print("  - Monitor precision, recall, and F1-score")
    else:
        print("â€¢ Regression-specific:")
        print("  - Use RMSE/MAE as primary metrics")
        print("  - Consider target transformation if skewed")
        print("  - Monitor residual patterns")
    
    print(f"\nğŸ“ˆ NEXT STEPS:")
    print("1. Implement preprocessing pipeline")
    print("2. Create baseline models")
    print("3. Feature selection and engineering")
    print("4. Hyperparameter tuning")
    print("5. Model ensembling")

generate_eda_summary_and_recommendations(
    train_df, test_df, target_analysis, feature_analysis, missing_analysis
)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("âœ… Libraries imported successfully!")



!pip install lightgbm


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# Identify target column
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
target_column = list(train_cols - test_cols)[0]

print(f"Dataset shapes:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Target column: {target_column}")

# Check target variable type
target_values = train_df[target_column]
is_classification = target_values.nunique() < 20 or target_values.dtype == 'object'

print(f"Problem type: {'Classification' if is_classification else 'Regression'}")
print(f"Target unique values: {target_values.nunique()}")

if is_classification:
    print(f"Class distribution:")
    print(target_values.value_counts())
else:
    print(f"Target statistics:")
    print(target_values.describe())



class MLPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.target_encoder = None
        self.numeric_features = []
        self.categorical_features = []
        
    def fit(self, X, y=None):
        """Fit preprocessors on training data"""
        
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        # Fit categorical encoders
        for col in self.categorical_features:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(X[col].astype(str))
        
        # Fit numeric scaler
        if self.numeric_features:
            self.scalers['numeric'] = RobustScaler()
            self.scalers['numeric'].fit(X[self.numeric_features])
        
        # Fit target encoder if needed
        if y is not None and (y.nunique() < 20 or y.dtype == 'object'):
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(y)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessors"""
        
        X_processed = X.copy()
        
        # Transform categorical features
        for col in self.categorical_features:
            if col in X_processed.columns:
                # Handle unseen categories
                X_processed[col] = X_processed[col].astype(str)
                mask = X_processed[col].isin(self.encoders[col].classes_)
                X_processed.loc[~mask, col] = self.encoders[col].classes_[0]
                X_processed[col] = self.encoders[col].transform(X_processed[col])
        
        # Transform numeric features
        if self.numeric_features and 'numeric' in self.scalers:
            X_processed[self.numeric_features] = self.scalers['numeric'].transform(
                X_processed[self.numeric_features]
            )
        
        return X_processed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def transform_target(self, y):
        """Transform target variable if needed"""
        if self.target_encoder is not None:
            return self.target_encoder.transform(y)
        return y
    
    def inverse_transform_target(self, y):
        """Inverse transform target variable if needed"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y)
        return y

print("âœ… Preprocessor class created!")



# Separate features and target
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

# Initialize and fit preprocessor
preprocessor = MLPreprocessor()
X_processed = preprocessor.fit_transform(X, y)
y_processed = preprocessor.transform_target(y)

# Process test data
X_test_processed = preprocessor.transform(test_df)

print(f"Processed training features shape: {X_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_processed, 
    test_size=0.2, 
    random_state=42,
    stratify=y_processed if is_classification else None
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")



def train_classification_models(X_train, y_train, X_val, y_val):
    """Train multiple classification models"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("Training Classification Models...")
    print("="*50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        results[name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'overfitting': train_acc - val_acc
        }
        
        trained_models[name] = model
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Overfitting: {train_acc - val_acc:.4f}")
    
    return results, trained_models

def train_regression_models(X_train, y_train, X_val, y_val):
    """Train multiple regression models"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("Training Regression Models...")
    print("="*50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        results[name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'overfitting_rmse': val_rmse - train_rmse
        }
        
        trained_models[name] = model
        
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}")
        print(f"Train RÂ²: {train_r2:.4f}")
        print(f"Val RÂ²: {val_r2:.4f}")
    
    return results, trained_models

print("âœ… Model training functions created!")



# Train models based on problem type
if is_classification:
    results, trained_models = train_classification_models(X_train, y_train, X_val, y_val)
    metric_name = 'val_accuracy'
    ascending = False
else:
    results, trained_models = train_regression_models(X_train, y_train, X_val, y_val)
    metric_name = 'val_rmse'
    ascending = True

# Create results DataFrame
results_df = pd.DataFrame(results).T
print(f"\nğŸ“Š MODEL COMPARISON:")
print("="*50)
print(results_df.round(4))

# Find best model
best_model_name = results_df.sort_values(metric_name, ascending=ascending).index[0]
best_model = trained_models[best_model_name]

print(f"\nğŸ�† BEST MODEL: {best_model_name}")



def perform_cross_validation(models, X_processed, y_processed, is_classification, cv_folds=5):
    """Perform cross-validation for all models"""
    
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = 'neg_root_mean_squared_error'
    
    cv_results = {}
    
    print(f"\nğŸ”„ CROSS-VALIDATION RESULTS ({cv_folds}-Fold):")
    print("="*50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_processed, y_processed, cv=cv, scoring=scoring)
        
        if not is_classification:
            scores = -scores  # Convert negative RMSE back to positive
        
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"{name}:")
        print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results

# Perform cross-validation
cv_results = perform_cross_validation(trained_models, X_processed, y_processed, is_classification)



def analyze_feature_importance(best_model, feature_names, top_n=15):
    """Analyze and visualize feature importance"""
    
    if hasattr(best_model, 'feature_importances_'):
        importance_scores = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance_scores = np.abs(best_model.coef_)
        if len(importance_scores.shape) > 1:
            importance_scores = importance_scores[0]
    else:
        print("Model doesn't support feature importance")
        return None
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(f"\nğŸ”� TOP {top_n} IMPORTANT FEATURES:")
    print("="*40)
    print(feature_importance.head(top_n).to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# Get feature names
feature_names = X.columns.tolist()

# Analyze feature importance
feature_importance = analyze_feature_importance(best_model, feature_names)



def detailed_model_evaluation(model, X_val, y_val, is_classification, model_name):
    """Detailed evaluation of the best model"""
    
    predictions = model.predict(X_val)
    
    print(f"\nğŸ“ˆ DETAILED EVALUATION - {model_name}")
    print("="*50)
    
    if is_classification:
        accuracy = accuracy_score(y_val, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    else:
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RÂ² Score: {r2:.4f}")
        
        # Residual plot
        residuals = y_val - predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Predicted vs Actual
        axes[0].scatter(y_val, predictions, alpha=0.6)
        axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs Actual')
        
        # Residual plot
        axes[1].scatter(predictions, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        plt.show()

# Detailed evaluation of best model
detailed_model_evaluation(best_model, X_val, y_val, is_classification, best_model_name)



def generate_predictions_and_submission(model, X_test_processed, preprocessor, sample_submission):
    """Generate predictions and create submission file"""
    
    print(f"\nğŸ�¯ GENERATING PREDICTIONS...")
    print("="*40)
    
    # Make predictions on test set
    test_predictions = model.predict(X_test_processed)
    
    # Inverse transform if needed
    if preprocessor.target_encoder is not None:
        test_predictions = preprocessor.inverse_transform_target(test_predictions)
    
    # Create submission DataFrame
    submission = sample_submission.copy()
    submission.iloc[:, 1] = test_predictions  # Assuming target is second column
    
    print(f"Predictions shape: {test_predictions.shape}")
    print(f"Submission shape: {submission.shape}")
    
    # Display sample predictions
    print(f"\nSample predictions:")
    print(submission.head(10))
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print(f"\nâœ… Submission file saved as 'submission.csv'")
    
    return submission, test_predictions

# Generate final predictions
submission, test_predictions = generate_predictions_and_submission(
    best_model, X_test_processed, preprocessor, sample_submission
)



import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
tf.random.set_seed(42)
print("TensorFlow version:", tf.__version__)



# Convert to float32 for better performance with neural networks
X_train_dl = X_train.astype(np.float32)
X_val_dl = X_val.astype(np.float32)
X_test_dl = X_test_processed.astype(np.float32)

if is_classification:
    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")



def create_neural_network(input_dim, is_classification, n_classes=None):
    """Create a neural network model"""
    
    model = models.Sequential([
        # Input layer
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
    ])
    
    # Output layer based on problem type
    if is_classification:
        if n_classes > 2:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(layers.Dense(1, activation='linear'))
        loss = 'mse'
        metrics = ['mae']
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics
    )
    
    return model

# Create the model
n_classes = len(np.unique(y_train)) if is_classification else None
dl_model = create_neural_network(X_train_dl.shape[1], is_classification, n_classes)

print("\nğŸ§  Neural Network Architecture:")
dl_model.summary()



# Setup callbacks for better training
dl_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nğŸš€ Training Neural Network...")
print("="*50)

# Train the model
dl_history = dl_model.fit(
    X_train_dl, y_train,
    validation_data=(X_val_dl, y_val),
    epochs=25,
    batch_size=32,
    callbacks=dl_callbacks,
    verbose=1
)


# Evaluate the model
print("\nğŸ“Š Neural Network Evaluation:")
print("="*40)

# Make predictions
train_pred_dl = dl_model.predict(X_train_dl, verbose=0)
val_pred_dl = dl_model.predict(X_val_dl, verbose=0)

if is_classification:
    if n_classes > 2:
        train_pred_classes = np.argmax(train_pred_dl, axis=1)
        val_pred_classes = np.argmax(val_pred_dl, axis=1)
    else:
        train_pred_classes = (train_pred_dl > 0.5).astype(int).flatten()
        val_pred_classes = (val_pred_dl > 0.5).astype(int).flatten()
    
    train_acc_dl = accuracy_score(y_train, train_pred_classes)
    val_acc_dl = accuracy_score(y_val, val_pred_classes)
    
    print(f"Neural Network Results:")
    print(f"Train Accuracy: {train_acc_dl:.4f}")
    print(f"Validation Accuracy: {val_acc_dl:.4f}")
    print(f"Overfitting: {train_acc_dl - val_acc_dl:.4f}")
    
else:
    train_rmse_dl = np.sqrt(mean_squared_error(y_train, train_pred_dl.flatten()))
    val_rmse_dl = np.sqrt(mean_squared_error(y_val, val_pred_dl.flatten()))
    
    print(f"Neural Network Results:")
    print(f"Train RMSE: {train_rmse_dl:.4f}")
    print(f"Validation RMSE: {val_rmse_dl:.4f}")
    print(f"Overfitting: {val_rmse_dl - train_rmse_dl:.4f}")



# Plot training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(dl_history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(dl_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Neural Network - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Metric plot
plt.subplot(1, 2, 2)
metric_name = 'accuracy' if is_classification else 'mae'
plt.plot(dl_history.history[metric_name], label=f'Training {metric_name.title()}', linewidth=2)
plt.plot(dl_history.history[f'val_{metric_name}'], label=f'Validation {metric_name.title()}', linewidth=2)
plt.title(f'Neural Network - {metric_name.title()}')
plt.xlabel('Epoch')
plt.ylabel(metric_name.title())
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Add neural network results to comparison
if is_classification:
    results['Neural Network'] = {
        'train_accuracy': train_acc_dl,
        'val_accuracy': val_acc_dl,
        'overfitting': train_acc_dl - val_acc_dl
    }
    comparison_metric = 'val_accuracy'
    ascending = False
else:
    results['Neural Network'] = {
        'train_rmse': train_rmse_dl,
        'val_rmse': val_rmse_dl,
        'train_r2': r2_score(y_train, train_pred_dl.flatten()),
        'val_r2': r2_score(y_val, val_pred_dl.flatten()),
        'overfitting_rmse': val_rmse_dl - train_rmse_dl
    }
    comparison_metric = 'val_rmse'
    ascending = True

# Update results DataFrame
results_df_updated = pd.DataFrame(results).T
print(f"\nğŸ“Š UPDATED MODEL COMPARISON (Including Neural Network):")
print("="*60)
print(results_df_updated.round(4))

# Find best model overall
best_model_overall = results_df_updated.sort_values(comparison_metric, ascending=ascending).index[0]
print(f"\nğŸ�† BEST MODEL OVERALL: {best_model_overall}")


# Make predictions with neural network
test_pred_dl = dl_model.predict(X_test_dl, verbose=0)

if is_classification:
    if n_classes > 2:
        test_pred_dl_final = np.argmax(test_pred_dl, axis=1)
    else:
        test_pred_dl_final = (test_pred_dl > 0.5).astype(int).flatten()
    
    # Inverse transform if needed
    if preprocessor.target_encoder is not None:
        test_pred_dl_final = preprocessor.inverse_transform_target(test_pred_dl_final)
else:
    test_pred_dl_final = test_pred_dl.flatten()

# Create neural network submission
submission_dl = sample_submission.copy()
submission_dl.iloc[:, 1] = test_pred_dl_final
submission_dl.to_csv('neural_network_submission.csv', index=False)

print(f"\nğŸ�¯ Neural Network Predictions Generated!")
print(f"Sample predictions: {test_pred_dl_final[:10]}")
print("âœ… Neural network submission saved as 'neural_network_submission.csv'")

# Compare with best ML model prediction
if best_model_overall != 'Neural Network':
    print(f"\nğŸ’¡ You can also ensemble {best_model_overall} with Neural Network for potentially better results!")



print("\n" + "="*60)
print("ğŸ�‰ COMPLETE MODEL COMPARISON SUMMARY")
print("="*60)

print(f"\nğŸ�† Best Overall Model: {best_model_overall}")

if is_classification:
    best_score = results_df_updated.loc[best_model_overall, 'val_accuracy']
    print(f"ğŸ“Š Best Validation Accuracy: {best_score:.4f}")
else:
    best_score = results_df_updated.loc[best_model_overall, 'val_rmse']
    print(f"ğŸ“Š Best Validation RMSE: {best_score:.4f}")

print(f"\nğŸ“� Generated Submissions:")
print(f"- ML Best Model: submission.csv")
print(f"- Neural Network: neural_network_submission.csv")

print(f"\nğŸš€ Next Steps:")
print("1. Try ensemble methods combining ML + DL")
print("2. Experiment with different neural network architectures")
print("3. Tune hyperparameters further")
print("4. Submit your best performing model!")





