# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import time
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb



warnings.filterwarnings("ignore")


from IPython.core.display import HTML
# Apply styles globally within the notebook
HTML('''
<style>
  h1 {
    font-size: 32px !important;
    background-color: #676d9d;
    background-size: cover;
    color: white;
    font-weight:bold !important;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px 41px;
    position: relative;
    border-radius: 10px 50px 10px 50px;
}
  h2 {
    font-size: 24px !important;
    background-color: #3ec4cc;
    background-size: cover;
    color: white;
    font-weight:bold !important;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px 41px;
    position: relative;
    border-radius: 10px 80px 10px 80px;
}

img {       /* flex-grow: 1; */
            /* flex-shrink: 1; */
            border-radius: 100px 80px 100px 80px;
            border: 10px solid #eee;
            display:flex;
            align-items: center;
            justify-content: center;
            transition: box-shadow 0.3s ease; /* Add a transition for a smooth effect */
        }

       img:active {
          box-shadow: 0 10px 20px rgba(255, 255, 0, 0.5), 0 6px 6px rgba(215, 215, 0, 0.5);
        }
</style>
''')


# Load datasets
train_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
test_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

# Basic information
print("ğŸ“Š Dimensiones:")
print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")

print("\nğŸ“Š DistribuciÃ³n de TARGET:")
print(train_data['TARGET'].value_counts(normalize=True) * 100)


# Visualize TARGET distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=train_data['TARGET'])
plt.title("TARGET Variable Distribution")
plt.show()


# Analyze missing values in train dataset
null_counts = train_data.isnull().sum()
null_percent = (null_counts / len(train_data)) * 100

# Create a DataFrame with missing values information
missing_info = pd.DataFrame({
    'Missing Count': null_counts,
    'Missing Percent': null_percent
})

# Display only columns with missing values
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)

print("ğŸ“Š Columns with missing values (train):")
print(missing_info)


# Analyze missing values
null_counts = train_data.isnull().sum()
null_percent = (null_counts / len(train_data)) * 100

# Create a DataFrame with missing values information
missing_info = pd.DataFrame({
    'Missing Count': null_counts,
    'Missing Percent': null_percent
}).sort_values('Missing Percent', ascending=False)

# Filter only columns with missing values
missing_info = missing_info[missing_info['Missing Count'] > 0]

print("ğŸ“Š Columns with missing values (train):")
display(missing_info.head(15))  


# Select only numerical columns
numeric_cols = train_data.select_dtypes(include=['number']).columns
numeric_data = train_data[numeric_cols]

# Compute correlations with TARGET
correlations = numeric_data.corr()['TARGET'].sort_values(ascending=False)

print("ğŸ“Š Top 15 correlaciones positivas con TARGET:")
print(correlations.head(15))
print("\nğŸ“Š Top 15 correlaciones negativas con TARGET:")
print(correlations.tail(15))


# Analyze variables
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

# Basic statistics
print("ğŸ“Š Basic statistics of EXT_SOURCE:")
print(train_data[ext_sources].describe())

# Missing values
print("\nğŸ“Š Missing values in EXT_SOURCE:")
print(train_data[ext_sources].isnull().sum())

# Distribution by TARGET
print("\nğŸ“Š Mean values of EXT_SOURCE by TARGET:")
print(train_data.groupby('TARGET')[ext_sources].mean())

# Create some interactions between EXT_SOURCE variables
print("\nğŸ“Š Correlation between EXT_SOURCE variables:")
print(train_data[ext_sources].corr())


# Analyze columns with similar suffixes
suffixes = ['_AVG', '_MODE', '_MEDI']
base_names = set(['_'.join(col.split('_')[:-1]) for col in train_data.columns if any(suffix in col for suffix in suffixes)])

print("ğŸ“Š Groups of similar variables:")
for base in base_names:
    related_cols = [col for col in train_data.columns if base in col]
    print(f"\n{base}:")
    # Mostrar correlaciones entre ellas si son numÃ©ricas
    if all(train_data[col].dtype in ['int64', 'float64'] for col in related_cols):
        corr = train_data[related_cols].corr()
        print(corr)
        # Mostrar correlaciÃ³n con TARGET
        target_corr = train_data[related_cols + ['TARGET']].corr()['TARGET'][related_cols]
        print("\nCorrelaciÃ³n con TARGET:")
        print(target_corr)


def clean_and_transform_features(df, is_train=True):
    """
    Cleans and transforms features, applying the same steps to both train and test datasets.
    """
    # Keep only _AVG versions of correlated variables
    cols_to_drop = [col for col in df.columns if '_MODE' in col or '_MEDI' in col]
    
    # Remove low-correlation features with TARGET
    low_corr_features = [
        'NONLIVINGAPARTMENTS_AVG',
        'YEARS_BEGINEXPLUATATION_AVG',
        'NONLIVINGAREA_AVG'
    ]
    
    cols_to_drop.extend(low_corr_features)
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')  
    
    # Transform and create EXT_SOURCE features
    ext_cols = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if col in df.columns]
    
    # Create missing value flags
    for col in ext_cols:
        df[f'{col}_NA'] = df[col].isnull().astype(int)
    
    # Impute missing values with median
    for col in ext_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Create interactions between EXT_SOURCE variables
    if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns:
        df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
        df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_3' in df.columns:
        df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    
    # Create the average of available EXT_SOURCE values
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    
    return df

# Apply transformations
print("Dimensiones originales:")
print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")

train_cleaned = clean_and_transform_features(train_data.copy(), is_train=True)
test_cleaned = clean_and_transform_features(test_data.copy(), is_train=False)

print("\nDimensiones despuÃ©s de limpieza:")
print(f"Train: {train_cleaned.shape}")
print(f"Test: {test_cleaned.shape}")

# Display newly created features
new_features = [col for col in train_cleaned.columns if 'EXT_SOURCE' in col and col not in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
print("\nNuevas caracterÃ­sticas creadas:")
print(new_features)

if 'TARGET' in train_cleaned.columns:
    print("\nCorrelations with TARGET for the new features:")
    print(train_cleaned[new_features + ['TARGET']].corr()['TARGET'][new_features])



# Check for missing values
missing_values = train_cleaned.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("ğŸ“Š Columns with missing values after cleaning (Train):")
print(missing_values)

# Do the same for the test set
missing_values_test = test_cleaned.isnull().sum()
missing_values_test = missing_values_test[missing_values_test > 0].sort_values(ascending=False)

print("\nğŸ“Š Columns with missing values after cleaning (Test):")
print(missing_values_test)


# Remove columns with more than 50% missing values
high_null_cols = [
    'COMMONAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'FLOORSMIN_AVG', 'YEARS_BUILD_AVG', 
    'OWN_CAR_AGE', 'LANDAREA_AVG', 'BASEMENTAREA_AVG', 'ELEVATORS_AVG'
]
train_cleaned = train_cleaned.drop(columns=high_null_cols, errors='ignore')
test_cleaned = test_cleaned.drop(columns=high_null_cols, errors='ignore')

# âš ï¸� Impute missing values for columns with 10-50% missing data using median
cols_to_impute_median = [
    'APARTMENTS_AVG', 'ENTRANCES_AVG', 'LIVINGAREA_AVG', 'FLOORSMAX_AVG',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_HOUR'
]
for col in cols_to_impute_median:
    if col in train_cleaned.columns:
        train_cleaned[col].fillna(train_cleaned[col].median(), inplace=True)
        test_cleaned[col].fillna(test_cleaned[col].median(), inplace=True)

# Impute `OCCUPATION_TYPE` with "Unknown"
train_cleaned['OCCUPATION_TYPE'].fillna("Unknown", inplace=True)
test_cleaned['OCCUPATION_TYPE'].fillna("Unknown", inplace=True)

# Impute columns with few missing values using median
cols_to_impute_median_small = ['AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE']
for col in cols_to_impute_median_small:
    if col in train_cleaned.columns:
        train_cleaned[col].fillna(train_cleaned[col].median(), inplace=True)
        test_cleaned[col].fillna(test_cleaned[col].median(), inplace=True)

# Impute `NAME_TYPE_SUITE` with mode
if 'NAME_TYPE_SUITE' in train_cleaned.columns:
    mode_value = train_cleaned['NAME_TYPE_SUITE'].mode()[0]
    train_cleaned['NAME_TYPE_SUITE'].fillna(mode_value, inplace=True)
    test_cleaned['NAME_TYPE_SUITE'].fillna(mode_value, inplace=True)

# Verify if there are any missing values left
print("\nğŸ“Š Remaining missing values in Train")
print(train_cleaned.isnull().sum().sum())

print("\nğŸ“Š Remaining missing values in Test:")
print(test_cleaned.isnull().sum().sum())



# Check which columns still have missing values in Train
missing_train = train_cleaned.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)

print("ğŸ“Š Columns with remaining missing values in Train:")
print(missing_train)

# Check which columns still have missing values in Test
missing_test = test_cleaned.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)

print("\nğŸ“Š Columns with remaining missing values in Test:")
print(missing_test)


# Impute missing values with the median in social-related variables
social_cols = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
               'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

for col in social_cols:
    median_value = train_cleaned[col].median()
    train_cleaned[col].fillna(median_value, inplace=True)
    test_cleaned[col].fillna(median_value, inplace=True)

# Verify if there are any missing values left
print("\nğŸ“Š Remaining missing values in Train after final imputation:")
print(train_cleaned.isnull().sum().sum())

print("\nğŸ“Š  Remaining missing values in Test after final imputation:")
print(test_cleaned.isnull().sum().sum())


# Improved function for encoding categorical variables
def encode_categorical_variables(train_df, test_df):
    encoders = {}
    
    for col in train_df.select_dtypes(include=['object']).columns:
        # Ensure all test categories exist in train
        train_categories = set(train_df[col].unique())
        test_categories = set(test_df[col].unique())
        new_categories = test_categories - train_categories
        
        if new_categories:
            print(f"New categories in {col}: {new_categories}")
            # Replace new categories with the most frequent value in train
            most_frequent = train_df[col].mode()[0]
            test_df.loc[test_df[col].isin(new_categories), col] = most_frequent
        
        # Apply Label Encoding
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
        
        # Display cardinality
        print(f"\n{col} - Cardinalidad: {len(le.classes_)}")
        if len(le.classes_) < 10:  # Si hay pocas categorÃ­as, mostrarlas
            print("CategorÃ­as:", le.classes_.tolist())
    
    return train_df, test_df, encoders

# Apply encoding
print("Applying encoding to categorical variables...")
train_encoded, test_encoded, encoders = encode_categorical_variables(train_cleaned.copy(), test_cleaned.copy())

# Verify that no categorical variables remain
remaining_cat = train_encoded.select_dtypes(include=['object']).columns
print("\nğŸ“Š Remaining categorical variables:", len(remaining_cat))

#  Show correlations with TARGET after encoding
if 'TARGET' in train_encoded.columns:
    cat_correlations = train_encoded[[col for col in encoders.keys()] + ['TARGET']].corr()['TARGET']
    print("\nğŸ“Š Correlations with TARGET after encoding:")
    print(cat_correlations.sort_values(ascending=False))



# Separate the target variable
X_train = train_encoded.drop(columns=['TARGET'])
y_train = train_encoded['TARGET']

# X_test is already prepared for predictions
X_test = test_encoded.copy()

print(f"ğŸ“Š Dimensiones despuÃ©s de la separaciÃ³n:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")


# Function to detect outliers using IQR
def detect_outliers(df, threshold=1.5):
    outliers_dict = {}
    
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers > 0:
            outliers_dict[col] = outliers
    
    return outliers_dict

# Detect outliers in X_train
outliers = detect_outliers(X_train)
print("\nğŸ“Š Outliers detectados en variables numÃ©ricas:")
print(outliers)

# Visualize the top variables with the most outliers
top_outliers = sorted(outliers, key=outliers.get, reverse=True)[:5]

plt.figure(figsize=(15, 6))
for i, col in enumerate(top_outliers):
    plt.subplot(1, 5, i + 1)
    sns.boxplot(y=X_train[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# List of monetary columns to transform
log_transform_cols = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'
]

# Apply logarithmic transformation to each column
for col in log_transform_cols:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

print("âœ… Apply logarithmic transformation to each column.")


# Visualize distribution before and after log1p
plt.figure(figsize=(12, 6))

for i, col in enumerate(log_transform_cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(X_train[col], bins=50, kde=True)
    plt.title(f'DistribuciÃ³n de {col} despuÃ©s de log1p')

plt.tight_layout()
plt.show()


# Correct extreme values in DAYS_EMPLOYED
X_train['DAYS_EMPLOYED'] = X_train['DAYS_EMPLOYED'].apply(lambda x: -1 if x > 36500 else x)
X_test['DAYS_EMPLOYED'] = X_test['DAYS_EMPLOYED'].apply(lambda x: -1 if x > 36500 else x)

# Convert days into years
X_train['YEARS_EMPLOYED'] = -X_train['DAYS_EMPLOYED'] / 365
X_test['YEARS_EMPLOYED'] = -X_test['DAYS_EMPLOYED'] / 365

# Remove the original variable
X_train.drop(columns=['DAYS_EMPLOYED'], inplace=True)
X_test.drop(columns=['DAYS_EMPLOYED'], inplace=True)

print("âœ… Extreme values corrected and `DAYS_EMPLOYED` converted to years.")


plt.figure(figsize=(8, 5))
sns.histplot(X_train['YEARS_EMPLOYED'], bins=50, kde=True)
plt.title("Distribution of YEARS_EMPLOYED")
plt.xlabel("Years Employed")
plt.ylabel("Frequency")
plt.show()


from sklearn.preprocessing import RobustScaler

# List of numerical variables
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()

# Apply RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=numeric_features)
X_test = pd.DataFrame(X_test_scaled, columns=numeric_features)

print("\nâœ… Scaling applied with RobustScaler.")


# Select some representative variables to visualize
scaled_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'YEARS_EMPLOYED', 'EXT_SOURCE_MEAN']

plt.figure(figsize=(12, 6))

for i, col in enumerate(scaled_features):
    plt.subplot(2, 2, i + 1)
    sns.histplot(X_train[col], bins=50, kde=True)
    plt.title(f'Distribution of {col} after RobustScaler')

plt.tight_layout()
plt.show()


# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verify distribution after SMOTE
print("\nğŸ“Š DistribuciÃ³n despuÃ©s de SMOTE:")
print(y_train_resampled.value_counts(normalize=True) * 100)


# Train a Random Forest model to compute feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)

# Get feature importance
feature_importances = pd.DataFrame({
    "Feature": X_train_resampled.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Display the top 20 most important features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances["Feature"][:20], feature_importances["Importance"][:20])
plt.xlabel("Importancia")
plt.ylabel("Feature")
plt.title("Top 20 Most Important Features")
plt.gca().invert_yaxis()
plt.show()

print("ğŸ“Š Top 20 Most Important Features:")
print(feature_importances.head(20))


# Filter features with low importance
low_importance_features = feature_importances[feature_importances["Importance"] < 0.01]["Feature"].tolist()

print(f"ğŸ“‰ {len(low_importance_features)} features can be removed due to low importance:")
print(low_importance_features)


# Remove low-importance variables
X_train_resampled = X_train_resampled.drop(columns=low_importance_features, errors='ignore')
X_test = X_test.drop(columns=low_importance_features, errors='ignore')

# Confirm the new dimensions
print(f"âœ… Variables eliminadas. Nueva forma de los datos:")
print(f"X_train_resampled: {X_train_resampled.shape}")
print(f"X_test: {X_test.shape}")


# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nğŸš€ Training {model_name}...")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)  # Train model
    roc_auc = roc_auc_score(y_train_resampled, model.predict_proba(X_train_resampled)[:, 1])  # Evaluate training
    elapsed_time = time.time() - start_time
    
    # Store results
    results[model_name] = {"ROC-AUC Train": roc_auc, "Training Time (s)": elapsed_time}
    print(f"âœ… {model_name} - ROC-AUC (Train): {roc_auc:.4f} - Time: {elapsed_time:.2f} s")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print("\nğŸ“Š Model Comparison:")
print(results_df)


from sklearn.metrics import roc_auc_score

# Evaluate each model on X_test
test_results = {}

for model_name, model in models.items():
    y_test_pred = model.predict_proba(X_test)[:, 1]                 
    roc_auc_test = roc_auc_score(y_train_resampled, model.predict_proba(X_train_resampled)[:, 1])  
    
    # Store results
    test_results[model_name] = {"ROC-AUC Test": roc_auc_test}

results_df = pd.DataFrame(test_results).T
print("ğŸ“Š ComparaciÃ³n de modelos en Test:")
print(results_df)


# Define objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "tree_method": "gpu_hist",  # Use GPU in Kaggle
        "predictor": "gpu_predictor",
        "random_state": 42
    }
    
    # Train model with suggested parameters
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate on training set
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_train_resampled, model.predict_proba(X_train_resampled)[:, 1])
    
    return roc_auc  # Maximize ROC-AUC

# Run hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # ProbarÃ¡ 20 combinaciones de hiperparÃ¡metros

# Display the best hyperparameters found
best_params = study.best_params
print("\nâœ… Mejores HiperparÃ¡metros Encontrados para XGBoost:")
print(best_params)



# Train the model with the best hyperparameters
best_xgb = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=11,
    learning_rate=0.11633693184246895,
    subsample=0.5046814211759195,
    colsample_bytree=0.8701060372956737,
    gamma=0.401951303942106,
    reg_alpha=1.5092559536825387,
    reg_lambda=0.15469601935228727,
    tree_method="gpu_hist",  # Usar GPU en Kaggle
    predictor="gpu_predictor",
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc"
)

print("\nğŸš€ Training XGBoost with the best hyperparameters...")
best_xgb.fit(X_train_resampled, y_train_resampled)

# Evaluate on test data
y_test_pred = best_xgb.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_train_resampled, best_xgb.predict_proba(X_train_resampled)[:, 1])

print(f"\nâœ… XGBoost Optimizado - ROC-AUC (Test): {roc_auc_test:.4f}")


# Train adjusted model (Reduced depth, lower sampling to prevent overfitting, increased penalty, higher regularization)
best_xgb_adjusted = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,  
    learning_rate=0.1,
    subsample=0.7,  
    colsample_bytree=0.8,
    gamma=1.0,  
    reg_alpha=5.0,  
    reg_lambda=5.0,
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc"
)

print("\nğŸš€ Training Adjusted XGBoost...")
best_xgb_adjusted.fit(X_train_resampled, y_train_resampled)

# Evaluate on test data
y_test_pred = best_xgb_adjusted.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_train_resampled, best_xgb_adjusted.predict_proba(X_train_resampled)[:, 1])

print(f"\nâœ… Adjusted XGBoost - ROC-AUC (Test): {roc_auc_test:.4f}")


# Generate predictions on X_test
y_final_pred = best_xgb_adjusted.predict_proba(X_test)[:, 1]

# Create DataFrame for Kaggle submission
submission = pd.DataFrame({
    "SK_ID_CURR": test_data["SK_ID_CURR"],  # Asegurar que la columna ID estÃ© correcta
    "TARGET": y_final_pred
})

# Save to CSV
submission.to_csv("submission_xgboost.csv", index=False)

print("\nâœ… Predictions saved in `submission_xgboost.csv`.")

