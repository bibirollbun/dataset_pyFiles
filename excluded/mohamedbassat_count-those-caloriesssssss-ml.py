

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set styling options
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)




print("Loading datasets...")
# Load the data
data_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Basic info about the datasets
print("\n--- Training Dataset Info ---")
data_train.info()
print("\n--- Test Dataset Info ---")
data_test.info()



print("\n--- Missing Values in Training Dataset ---")
print(data_train.isnull().sum())
print("\n--- Missing Values in Test Dataset ---")
print(data_test.isnull().sum())




print("\n--- Training Dataset Statistics ---")
print(data_train.describe())
print("\n--- Test Dataset Statistics ---")
print(data_test.describe())




# Let's check the first few rows of both datasets
print("\n--- First 5 rows of Training Dataset ---")
print(data_train.head())
print("\n--- First 5 rows of Test Dataset ---")
print(data_test.head())



print(f"\nDuplicate rows in training dataset: {data_train.duplicated().sum()}")
print(f"Duplicate rows in test dataset: {data_test.duplicated().sum()}")




print("\n--- Sex Distribution in Training Dataset ---")
print(data_train['Sex'].value_counts())
print("\n--- Sex Distribution in Test Dataset ---")
print(data_test['Sex'].value_counts())



def plot_distributions(df, title):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'id' in numeric_features:
        numeric_features.remove('id')
    
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel(feature)
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Plot distributions for both datasets
print("\n--- Feature Distributions ---")
plot_distributions(data_train, 'Training Dataset Feature Distributions')
plot_distributions(data_test, 'Test Dataset Feature Distributions')



def compare_distributions(train_df, test_df):
    #Compare feature distributions between train and test sets
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'id' in numeric_features:
        numeric_features.remove('id')
    if 'Calories' in numeric_features:
        numeric_features.remove('Calories')  # Calories is not in test dataset
    
    n_features = len(numeric_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        sns.kdeplot(train_df[feature], ax=axes[i], label='Train', fill=True, alpha=0.5)
        sns.kdeplot(test_df[feature], ax=axes[i], label='Test', fill=True, alpha=0.5)
        axes[i].set_title(f'{feature} Distribution: Train vs Test')
        axes[i].set_xlabel(feature)
        axes[i].legend()
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Distributions: Training vs Test Dataset', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Compare distributions
print("\n--- Comparing Feature Distributions: Train vs Test ---")
compare_distributions(data_train, data_test)



# Age distribution by Sex
plt.figure(figsize=(12, 6))
sns.histplot(data=data_train, x='Age', hue='Sex', multiple='stack', bins=20)
plt.title('Age Distribution by Sex', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()




def plot_feature_vs_target(df, target='Calories'):
    """Plot the relationship between each feature and the target variable"""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'id' in numeric_features:
        numeric_features.remove('id')
    if target in numeric_features:
        numeric_features.remove(target)
    
    n_features = len(numeric_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        # Scatter plot with regression line
        sns.regplot(x=feature, y=target, data=df, ax=axes[i], scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        axes[i].set_title(f'{feature} vs {target}')
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Relationship Between Features and {target}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Drop 'id' and 'Sex' before computing correlations
cols_to_drop = ['id', 'Sex']
corr_matrix = data_train.drop(cols_to_drop, axis=1).corr()

print("\n--- Correlation Matrix (without Sex) ---")
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    linewidths=0.5
)
plt.title("Feature Correlation Matrix (excluding Sex)", fontsize=16)
plt.tight_layout()
plt.show()

# Sort correlations with the target variable
calories_correlations = corr_matrix['Calories'].sort_values(ascending=False)
print("\n--- Feature Correlations with Calories (excluding Sex) ---")
print(calories_correlations)



def plot_boxplots(df):
    """Plot boxplots to identify outliers in numerical features"""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'id' in numeric_features:
        numeric_features.remove('id')
    
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        sns.boxplot(y=df[feature], ax=axes[i])
        axes[i].set_title(f'{feature} Boxplot')
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Boxplots for Outlier Detection', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Plot boxplots
print("\n--- Boxplots for Outlier Detection ---")
plot_boxplots(data_train)

# Let's create a function to detect outliers using IQR method
def detect_outliers(df, feature):
    """Detect outliers using IQR method"""
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers.shape[0], lower_bound, upper_bound

# Check for outliers in each numeric feature
print("\n--- Outlier Detection using IQR Method ---")
numeric_features = data_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numeric_features:
    numeric_features.remove('id')

for feature in numeric_features:
    outlier_count, lower, upper = detect_outliers(data_train, feature)
    print(f"{feature}: {outlier_count} outliers detected. Bounds: [{lower:.2f}, {upper:.2f}]")



print("\n--- Feature Engineering ---")

# Create a function for feature engineering
def engineer_features(df):
    """Create new features based on existing ones"""
    df_new = df.copy()
    
    # Encode Sex
    df_new['Sex_Encoded'] = df_new['Sex'].map({'male': 1, 'female': 0})
    
    # Create BMI
    df_new['BMI'] = df_new['Weight'] / ((df_new['Height'] / 100) ** 2)
    
    # Create Age Groups
    df_new['Age_Group'] = pd.cut(
        df_new['Age'], bins=[0, 25, 35, 45, 60, 100],
        labels=['18-25', '26-35', '36-45', '46-60', '60+']
    )
    
    # Intensity metrics
    df_new['Workout_Intensity'] = df_new['Heart_Rate'] / df_new['Age']
    df_new['Estimated_MET'] = df_new['Heart_Rate'] / (df_new['Age'] * 0.7)
    
    # Temperature difference
    df_new['Body_Temp_Diff'] = df_new['Body_Temp'] - 37.0
    
    # Duration ratios
    df_new['Duration_per_kg'] = df_new['Duration'] / df_new['Weight']
    df_new['HR_Duration_Ratio'] = df_new['Heart_Rate'] / df_new['Duration']
    
    return df_new

# Apply feature engineering
train_engineered = engineer_features(data_train)
test_engineered  = engineer_features(data_test)

# Preview engineered features
print("\n--- Preview of Engineered Features (Train) ---")
print(train_engineered.head()[[
    'Sex', 'Sex_Encoded', 'Age', 'Age_Group', 'Height', 'Weight',
    'BMI', 'Duration', 'Heart_Rate', 'Workout_Intensity',
    'Estimated_MET', 'Body_Temp', 'Body_Temp_Diff'
]])

# Compute correlations only on numeric columns
df_numeric = train_engineered.select_dtypes(include='number')
corr_matrix = df_numeric.corr()

# Rank features by absolute correlation with Calories
corr_with_cal = corr_matrix['Calories'].abs().sort_values(ascending=False)

top10 = corr_with_cal.drop('Calories').head(10).index.tolist()
print("Top 10 features most correlated with Calories:")
for feat in top10:
    print(f"  {feat}: {corr_with_cal[feat]:.3f}")

# Visualize correlations among top features and Calories
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_numeric[top10 + ['Calories']].corr(),
    annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5
)
plt.title("Correlation Matrix: Top 10 Features vs. Calories", fontsize=14)
plt.tight_layout()
plt.show()

# Subset dataset for modeling
train_selected = train_engineered[top10 + ['Calories']].copy()
test_selected  = test_engineered[top10].copy()

print("\nSelected train shape:", train_selected.shape)
print("Selected test shape: ",  test_selected.shape)



# ML Model with Optuna Hyperparameter Tuning for Calorie Prediction
# Playground Series Season 5, Episode 5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
# Load the data
data_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print(f"Training data shape: {data_train.shape}")
print(f"Testing data shape: {data_test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")




# Feature Engineering Function
def engineer_features(df):
    """Create new features based on the highest-correlated transformations"""
    df_new = df.copy()
    
    # Keep core numeric columns: Duration and Heart_Rate are original
    # Encode Sex (optional, original Sex not directly used)
    df_new['Sex_Encoded'] = df_new['Sex'].map({'male': 1, 'female': 0})
    
    # Only include transformations with strong correlation to Calories
    df_new['BMI'] = df_new['Weight'] / ((df_new['Height'] / 100) ** 2)
    df_new['Workout_Intensity'] = df_new['Heart_Rate'] / df_new['Age']
    df_new['Estimated_MET'] = df_new['Heart_Rate'] / (df_new['Age'] * 0.7)
    df_new['Body_Temp_Diff'] = df_new['Body_Temp'] - 37.0
    df_new['Duration_per_kg'] = df_new['Duration'] / df_new['Weight']
    df_new['HR_Duration_Ratio'] = df_new['Heart_Rate'] / df_new['Duration']

    return df_new



# Preprocess data
def preprocess_data(df, is_train=True):
    """
    Preprocess the data by handling outliers, scaling features, and preparing for modeling
    """
    # Get engineered features
    df_processed = engineer_features(df)
    
    # Handle outliers using capping
    def cap_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    # Cap outliers for specific features
    numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    for feature in numeric_features:
        df_processed = cap_outliers(df_processed, feature)
    
    # Select features to keep (dropping original categorical and id)
    df_processed = df_processed.drop(['id', 'Sex'], axis=1)
    
    return df_processed

print("\nPreprocessing data...")
# Apply preprocessing
train_processed = preprocess_data(data_train)
test_processed = preprocess_data(data_test, is_train=False)




from sklearn.model_selection import train_test_split

# 1. Define features and target
X = train_processed.drop('Calories', axis=1)
y = train_processed['Calories']

# 2. Split into train / validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,      # 20% held out for validation
    random_state=42,     # for reproducibility
    shuffle=True
)

# 3. Confirm shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_valid shape: {y_valid.shape}")



X_train.info()


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Define models to compare
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5),

}

# RMSE scorer (note: negative because cross_val_score treats higher = better)
rmse_scorer = make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
                          greater_is_better=False)

# Cross-validation function with tqdm progress bar
def evaluate_models(X, y, models, cv=5):
    results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    print("\n--- Evaluating Models ---")
    for name in tqdm(models, desc="Evaluating"):
        model = models[name]
        scores = cross_val_score(model, X, y, scoring=rmse_scorer, cv=kf)
        rmse_scores = -scores  # make positive
        results[name] = {
            'RMSE Mean': np.mean(rmse_scores),
            'RMSE Std': np.std(rmse_scores)
        }
        print(f"{name}: RMSE = {results[name]['RMSE Mean']:.4f} (+/- {results[name]['RMSE Std']:.4f})")
    return results

# Run evaluation
results = evaluate_models(X, y, models)

# Determine best model
best_model_name = min(results, key=lambda k: results[k]['RMSE Mean'])
best_model = models[best_model_name]
print(f"\nTraining final model: {best_model_name}")
best_model.fit(X, y)

# Plot RMSE scores
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
rmse_means = [results[name]['RMSE Mean'] for name in model_names]
rmse_stds = [results[name]['RMSE Std'] for name in model_names]

plt.bar(model_names, rmse_means, yerr=rmse_stds, capsize=5, color='skyblue')
plt.ylabel("RMSE")
plt.title("Model Comparison (5-Fold CV)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




print("\nMaking predictions on test set...")
# Apply to test set
test_preds = best_model.predict(test_processed)
print("\nFinal predictions on test set generated.")
test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

test_ids = test['id']  # or whatever your raw test DataFrame is named

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_preds
})

print("\nSubmission dataframe:")
print(submission.head())
print(f"Submission shape: {submission.shape}")


# Save submission file
submission.to_csv('forest_optuna_submission.csv', index=False)
print("\nSubmission file saved as 'forest_optuna_submission.csv'")

# Display submission statistics
print("\nSubmission Statistics:")
print(f"  Min predicted calories: {submission['Calories'].min():.2f}")
print(f"  Max predicted calories: {submission['Calories'].max():.2f}")
print(f"  Mean predicted calories: {submission['Calories'].mean():.2f}")
print(f"  Median predicted calories: {submission['Calories'].median():.2f}")

print("\nModel training and prediction completed successfully!")




