import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")



delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
not_delay_4_6 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
not_delay_7_9 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
pilot_10 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')
sample_solution = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/sample_Solution.csv')

print("Dataset shapes:")
print(f"Delay 4-6: {delay_4_6.shape}")
print(f"Not Delay 4-6: {not_delay_4_6.shape}")
print(f"Delay 7-9: {delay_7_9.shape}")
print(f"Not Delay 7-9: {not_delay_7_9.shape}")
print(f"Pilot 10: {pilot_10.shape}")
print(f"Sample Solution: {sample_solution.shape}")


# Add label column to datasets
delay_4_6['label'] = 1
not_delay_4_6['label'] = 0
delay_7_9['label'] = 1
not_delay_7_9['label'] = 0

# Combine all training data
train_data = pd.concat([
    delay_4_6,
    not_delay_4_6,
    delay_7_9,
    not_delay_7_9
], ignore_index=True)

print(f"Combined training data shape: {train_data.shape}")
print(f"Label distribution:")
print(train_data['label'].value_counts())
print(f"Label distribution (%):") 
print(train_data['label'].value_counts(normalize=True) * 100)


# Basic information about the dataset
print("Training Data Info:")
print(train_data.info())
print("\nFirst few rows:")
print(train_data.head())
print("\nBasic statistics:")
print(train_data.describe())


# Check missing values
missing_values = train_data.isnull().sum()
missing_percent = (missing_values / len(train_data)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percent': missing_percent.values
})

missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)

print("Missing values summary:")
print(missing_df)

# Visualize missing values
if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=missing_df.head(10), x='Missing_Percent', y='Column')
    plt.title('Top 10 Columns with Missing Values (%)')
    plt.xlabel('Missing Percentage')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(train_data.isnull().head(100), cbar=True, yticklabels=False)
    plt.title('Missing Values Heatmap (First 100 rows)')
    plt.tight_layout()
    plt.show()


# Analyze label distribution
plt.figure(figsize=(15, 10))

# Label distribution
plt.subplot(2, 3, 1)
sns.countplot(data=train_data, x='label')
plt.title('Label Distribution')
plt.xlabel('Label (0: No Delay, 1: Delay)')

# Distribution by Ship Mode
if 'Ship Mode' in train_data.columns:
    plt.subplot(2, 3, 2)
    sns.countplot(data=train_data, x='Ship Mode', hue='label')
    plt.title('Delay by Ship Mode')
    plt.xticks(rotation=45)

# Distribution by Stock class
if 'Stock class' in train_data.columns:
    plt.subplot(2, 3, 3)
    sns.countplot(data=train_data, x='Stock class', hue='label')
    plt.title('Delay by Stock Class')

# Distribution by SUBSIDIARY_CD
if 'SUBSIDIARY_CD' in train_data.columns:
    plt.subplot(2, 3, 4)
    top_subsidiaries = train_data['SUBSIDIARY_CD'].value_counts().head(5).index
    subset = train_data[train_data['SUBSIDIARY_CD'].isin(top_subsidiaries)]
    sns.countplot(data=subset, x='SUBSIDIARY_CD', hue='label')
    plt.title('Delay by Top 5 Subsidiaries')
    plt.xticks(rotation=45)

# SO QTY distribution
if 'SO QTY' in train_data.columns:
    plt.subplot(2, 3, 5)
    train_data.boxplot(column='SO QTY', by='label', ax=plt.gca())
    plt.title('SO QTY Distribution by Label')
    plt.suptitle('')

# SUPPLIER INV AMOUNT distribution
if 'SUPPLIER INV AMOUNT' in train_data.columns:
    plt.subplot(2, 3, 6)
    train_data.boxplot(column='SUPPLIER INV AMOUNT', by='label', ax=plt.gca())
    plt.title('Supplier Inv Amount by Label')
    plt.suptitle('')

plt.tight_layout()
plt.show()


def preprocess_data(df, is_training=True):
    """
    Preprocess the data for machine learning
    """
    df_processed = df.copy()
    
    # Convert Order date to datetime with mixed format handling
    if 'Order date' in df_processed.columns:
        df_processed['Order date'] = pd.to_datetime(df_processed['Order date'], format='mixed', errors='coerce')
        # Extract date features
        df_processed['order_month'] = df_processed['Order date'].dt.month
        df_processed['order_day'] = df_processed['Order date'].dt.day
        df_processed['order_dayofweek'] = df_processed['Order date'].dt.dayofweek
        df_processed['order_quarter'] = df_processed['Order date'].dt.quarter
    
    # Convert VSD to datetime if exists
    if 'VSD' in df_processed.columns:
        df_processed['VSD'] = pd.to_datetime(df_processed['VSD'], format='mixed', errors='coerce')
        # Calculate days between order and VSD
        if 'Order date' in df_processed.columns:
            df_processed['days_to_vsd'] = (df_processed['VSD'] - df_processed['Order date']).dt.days
    
    # Handle categorical variables with high cardinality
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Remove datetime and ID columns from categorical processing
    categorical_cols = [col for col in categorical_cols if col not in ['Order date', 'VSD', 'ID']]
    
    # For high cardinality categorical variables, use frequency encoding
    high_cardinality_cols = []
    for col in categorical_cols:
        if df_processed[col].nunique() > 50:
            high_cardinality_cols.append(col)
            # Frequency encoding
            freq_map = df_processed[col].value_counts().to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map).fillna(0)
    
    # Drop original high cardinality columns
    df_processed = df_processed.drop(columns=high_cardinality_cols)
    
    # Update categorical columns list
    categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
    
    # Handle remaining categorical variables with Label Encoding
    le_dict = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            # Handle missing values
            df_processed[col] = df_processed[col].fillna('Unknown')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    # Handle numerical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove label from numerical columns if it exists
    if 'label' in numerical_cols:
        numerical_cols.remove('label')
    
    # Fill missing values in numerical columns
    imputer = SimpleImputer(strategy='median')
    df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
    
    # Create additional features
    if 'SO QTY' in df_processed.columns and 'ALLOCATION QTY' in df_processed.columns:
        df_processed['qty_ratio'] = df_processed['ALLOCATION QTY'] / (df_processed['SO QTY'] + 1)
    
    if 'SUPPLIER INV AMOUNT' in df_processed.columns and 'SO QTY' in df_processed.columns:
        df_processed['price_per_unit'] = df_processed['SUPPLIER INV AMOUNT'] / (df_processed['SO QTY'] + 1)
    
    # Drop datetime columns as they're not needed for modeling
    datetime_cols = ['Order date', 'VSD']
    df_processed = df_processed.drop(columns=[col for col in datetime_cols if col in df_processed.columns])
    
    return df_processed, le_dict

# Preprocess training data
print("Preprocessing training data...")
X_processed, label_encoders = preprocess_data(train_data, is_training=True)
y = X_processed['label']
X = X_processed.drop(columns=['label'])

print(f"Processed training data shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Features: {list(X.columns)[:10]}...")  # Show first 10 features


import numpy as np
import pandas as pd
import hashlib

def hash_series(series):
    """Táº¡o mÃ£ hash duy nháº¥t cho má»™t Series Ä‘á»ƒ kiá»ƒm tra trÃ¹ng láº·p nhanh."""
    return hashlib.md5(pd.util.hash_pandas_object(series, index=False).values).hexdigest()

# 1. Remove duplicate columns
print("Checking for duplicate columns...")

# Táº¡o hash cho tá»«ng cá»™t Ä‘á»ƒ so sÃ¡nh
col_hashes = {col: hash_series(X[col]) for col in X.columns}
reverse_map = {}
for col, h in col_hashes.items():
    reverse_map.setdefault(h, []).append(col)

# Láº¥y ra cÃ¡c cá»™t trÃ¹ng láº·p (chá»‰ giá»¯ láº¡i má»™t cá»™t Ä‘áº§u tiÃªn má»—i nhÃ³m)
duplicate_cols = [col for group in reverse_map.values() if len(group) > 1 for col in group[1:]]

if duplicate_cols:
    print(f"Found and removed {len(duplicate_cols)} duplicate columns:")
    print(duplicate_cols)
    X.drop(columns=duplicate_cols, inplace=True)
else:
    print("No duplicate columns found.")

# 2. Remove highly correlated features
print("\nChecking for highly correlated features (> 0.95)...")

# Chá»‰ láº¥y cá»™t sá»‘ Ä‘á»ƒ tÃ­nh correlation
num_cols = X.select_dtypes(include=[np.number])
corr_matrix = num_cols.corr().abs()

# Tam giÃ¡c trÃªn
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Láº¥y cÃ¡c cá»™t cÃ³ tÆ°Æ¡ng quan > 0.95
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

if to_drop:
    print(f"Found and removed {len(to_drop)} highly correlated columns:")
    print(to_drop)
    X.drop(columns=to_drop, inplace=True)
else:
    print("No highly correlated features found.")

print(f"\nâœ… Shape of training data after cleanup: {X.shape}")



# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Data scaling completed!")
print(f"Scaled data shape: {X_scaled.shape}")

# Feature correlation analysis
plt.figure(figsize=(20, 16))
corr_matrix = X_scaled.corrwith(y).abs().sort_values(ascending=False)

# Plot top 20 features with highest correlation to target
top_features = corr_matrix.head(20)
plt.subplot(2, 2, 1)
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top 20 Features Correlation with Target')
plt.xlabel('Absolute Correlation')

# Plot correlation heatmap for top features
top_feature_names = top_features.index[:15]
subset_data = pd.concat([X_scaled[top_feature_names], y], axis=1)
plt.subplot(2, 2, 2)
sns.heatmap(subset_data.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap - Top Features')

# Feature importance using Random Forest
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_scaled, y)
feature_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

plt.subplot(2, 2, 3)
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.xlabel('Importance')

# Distribution of target variable
plt.subplot(2, 2, 4)
sns.countplot(x=y)
plt.title('Target Distribution')
plt.xlabel('Label (0: No Delay, 1: Delay)')

plt.tight_layout()
plt.show()

print("\nTop 10 most important features:")
print(feature_importance.head(10))


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# 1. Split dá»¯ liá»‡u
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Label distribution in training set:\n{y_train.value_counts(normalize=True)}")

# 2. Ä�á»‹nh nghÄ©a 2 mÃ´ hÃ¬nh
models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',  # GPU-friendly (má»›i)
        device='cuda',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        device='gpu'
    )
}

# 3. Train vÃ  Ä‘Ã¡nh giÃ¡
model_results = {}

for name, model in models.items():
    print(f"\nğŸš€ Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,  # <-- cáº§n dÃ²ng nÃ y
        'y_pred_proba': y_pred_proba
    }
    
    print(f"âœ… Accuracy: {accuracy:.4f}")
    print(f"âœ… AUC: {auc_score:.4f}")
    print(f"âœ… CV AUC: {cv_scores.mean():.4f} Â± {cv_scores.std()*2:.4f}")

# 4. Chá»�n mÃ´ hÃ¬nh tá»‘t nháº¥t
best_model_name = max(model_results, key=lambda x: model_results[x]['auc'])
best_model = model_results[best_model_name]['model']
print(f"\nğŸ�¯ Best model: {best_model_name} (AUC = {model_results[best_model_name]['auc']:.4f})")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Táº¡o báº£ng so sÃ¡nh káº¿t quáº£
results_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [results['accuracy'] for results in model_results.values()],
    'AUC Score': [results['auc'] for results in model_results.values()],
    'CV Mean': [results['cv_mean'] for results in model_results.values()],
    'CV Std': [results['cv_std'] for results in model_results.values()]
})

print("ğŸ“Š Model Performance Comparison:")
print(results_df.round(4))

# 2. Váº½ biá»ƒu Ä‘á»“ so sÃ¡nh
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Biá»ƒu Ä‘á»“ Accuracy
axes[0, 0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
axes[0, 0].set_title('ğŸ”� Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)

# Biá»ƒu Ä‘á»“ AUC
axes[0, 1].bar(results_df['Model'], results_df['AUC Score'], color='lightgreen')
axes[0, 1].set_title('ğŸ’¯ AUC Score Comparison')
axes[0, 1].set_ylabel('AUC Score')
axes[0, 1].tick_params(axis='x', rotation=45)

# Biá»ƒu Ä‘á»“ Cross-validation AUC
axes[1, 0].bar(results_df['Model'], results_df['CV Mean'], yerr=results_df['CV Std'], 
               color='lightcoral', capsize=5)
axes[1, 0].set_title('ğŸ“‰ CV AUC Score (Â± 2 Std)')
axes[1, 0].set_ylabel('CV AUC Score')
axes[1, 0].tick_params(axis='x', rotation=45)

# Confusion Matrix cá»§a best model
best_model_name = results_df.loc[results_df['AUC Score'].idxmax(), 'Model']
best_model_results = model_results[best_model_name]

cm = confusion_matrix(y_test, best_model_results['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title(f'ğŸ§® Confusion Matrix - {best_model_name}')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 3. In káº¿t quáº£ tá»•ng há»£p
print(f"\nğŸ�† Best model: {best_model_name}")
print(f"â­� AUC Score: {best_model_results['auc']:.4f}")

# 4. Classification report chi tiáº¿t
print(f"\nğŸ§¾ Classification Report for {best_model_name}:")
print(classification_report(y_test, best_model_results['y_pred']))



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

print(f"ğŸ�¯ Hyperparameter tuning for {best_model_name}...")

best_model = model_results[best_model_name]['model']

# Chá»‰ Ä‘á»‹nh grid search cho 2 model
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
elif best_model_name == 'LightGBM':
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31]
    }

# Grid Search
grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nâœ… Best parameters: {grid_search.best_params_}")
print(f"ğŸ“ˆ Best CV AUC: {grid_search.best_score_:.4f}")

# Evaluate tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
y_pred_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_auc = roc_auc_score(y_test, y_pred_proba_tuned)

print(f"\nğŸ”� Tuned model performance:")
print(f"Accuracy: {tuned_accuracy:.4f}")
print(f"AUC Score: {tuned_auc:.4f}")
print(f"ğŸ“Š AUC Improvement: {tuned_auc - best_model_results['auc']:.4f}")

# LÆ°u láº¡i mÃ´ hÃ¬nh tá»‘t nháº¥t
final_model = tuned_model



def preprocess_test_data(df, label_encoders, scaler):
    """
    Preprocess test data using the same transformations as training data
    """
    df_processed = df.copy()
    
    # Convert Order date to datetime
    if 'Order date' in df_processed.columns:
        df_processed['Order date'] = pd.to_datetime(df_processed['Order date'])
        # Extract date features
        df_processed['order_month'] = df_processed['Order date'].dt.month
        df_processed['order_day'] = df_processed['Order date'].dt.day
        df_processed['order_dayofweek'] = df_processed['Order date'].dt.dayofweek
        df_processed['order_quarter'] = df_processed['Order date'].dt.quarter
    
    # Convert VSD to datetime if exists
    if 'VSD' in df_processed.columns:
        df_processed['VSD'] = pd.to_datetime(df_processed['VSD'], errors='coerce')
        # Calculate days between order and VSD
        if 'Order date' in df_processed.columns:
            df_processed['days_to_vsd'] = (df_processed['VSD'] - df_processed['Order date']).dt.days
    
    # Handle categorical variables with high cardinality (frequency encoding)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Order date', 'VSD', 'ID']]
    
    # Apply frequency encoding for high cardinality columns
    # Note: In a real scenario, we should save frequency maps from training data
    high_cardinality_cols = []
    for col in categorical_cols:
        if col in df_processed.columns and df_processed[col].nunique() > 50:
            high_cardinality_cols.append(col)
            # For test data, use a simple approach or saved frequency maps
            freq_map = df_processed[col].value_counts().to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map).fillna(0)
    
    # Drop original high cardinality columns
    df_processed = df_processed.drop(columns=high_cardinality_cols)
    
    # Update categorical columns list
    categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
    
    # Handle remaining categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
            # For test data, handle unseen categories
            try:
                # This is a simplified approach - in practice, save encoders from training
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            except:
                df_processed[col] = 0  # Default value for unseen categories
    
    # Handle numerical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values in numerical columns
    imputer = SimpleImputer(strategy='median')
    df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
    
    # Create additional features
    if 'SO QTY' in df_processed.columns and 'ALLOCATION QTY' in df_processed.columns:
        df_processed['qty_ratio'] = df_processed['ALLOCATION QTY'] / (df_processed['SO QTY'] + 1)
    
    if 'SUPPLIER INV AMOUNT' in df_processed.columns and 'SO QTY' in df_processed.columns:
        df_processed['price_per_unit'] = df_processed['SUPPLIER INV AMOUNT'] / (df_processed['SO QTY'] + 1)
    
    # Drop datetime columns
    datetime_cols = ['Order date', 'VSD']
    df_processed = df_processed.drop(columns=[col for col in datetime_cols if col in df_processed.columns])
    
    # Keep only features that exist in training data
    common_features = [col for col in X.columns if col in df_processed.columns]
    missing_features = [col for col in X.columns if col not in df_processed.columns]
    
    if missing_features:
        print(f"Missing features in test data: {missing_features}")
        # Add missing features with default values
        for feature in missing_features:
            df_processed[feature] = 0
    
    # Select and reorder features to match training data
    df_processed = df_processed[X.columns]
    
    return df_processed

# Preprocess test data (PILOT_10.csv)
print("Preprocessing test data...")
test_data_processed = preprocess_test_data(pilot_10, label_encoders, scaler)

# Scale test data
test_data_scaled = scaler.transform(test_data_processed)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=X.columns)

print(f"Test data shape: {test_data_scaled.shape}")
print("Test data preprocessing completed!")


# Make predictions on test data
print("Making predictions on test data...")

predictions = final_model.predict(test_data_scaled)
prediction_probabilities = final_model.predict_proba(test_data_scaled)[:, 1]

print(f"Predictions made for {len(predictions)} samples")
print(f"Prediction distribution:")
print(pd.Series(predictions).value_counts())
print(f"Prediction distribution (%):")
print(pd.Series(predictions).value_counts(normalize=True) * 100)

# Create submission file
submission = pd.DataFrame({
    'ID': pilot_10['ID'],
    'label': predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print("First 10 predictions:")
print(submission.head(10))

# Verify submission format matches sample
print(f"\nSubmission shape: {submission.shape}")
print(f"Sample solution shape: {sample_solution.shape}")
print(f"Columns match: {list(submission.columns) == list(sample_solution.columns)}")

