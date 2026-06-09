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


# Comprehensive code for South African Job Application Success Prediction

# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# 1. DATA LOADING AND INITIAL EXPLORATION
print("1. LOADING DATA...")
print("="*50)

# Load the data
train_data = pd.read_csv('/kaggle/input/south-african-opportunity-prediction-challenge/train_sample.csv')
test_data = pd.read_csv('/kaggle/input/south-african-opportunity-prediction-challenge/test_sample.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"\nTarget distribution:")
print(train_data['Progress'].value_counts().sort_index())
print(f"\nSuccess rate: {train_data['Successful'].mean():.2%}")

# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
print("\n2. FEATURE ENGINEERING...")
print("="*50)

def preprocess_data(df, is_train=True):
   """Comprehensive preprocessing and feature engineering"""
   
   # Create a copy to avoid modifying original
   data = df.copy()
   
   # 2.1 Handle Disciplines (multi-label encoding)
   def to_disciplines_list(x):
       if isinstance(x, list):
           return [d.strip() for d in x if d.strip()]
       elif pd.isna(x) or x == '':
           return []
       else:
           return [d.strip() for d in str(x).split(',') if d.strip()]
   
   data['Disciplines_list'] = data['Disciplines'].apply(to_disciplines_list)
   
   # Count number of disciplines
   data['num_disciplines'] = data['Disciplines_list'].apply(len)
   
   # 2.2 Create binary features
   data['has_disciplines'] = (data['num_disciplines'] > 0).astype(int)
   data['is_matric'] = (data['Qualification'] == 'National Senior Certificate').astype(int)
   data['has_degree'] = (~data['is_matric']).astype(int)
   
   # 2.3 Age-based features
   data['age_group'] = pd.cut(data['Age'], 
                               bins=[0, 20, 25, 30, 100], 
                               labels=['very_young', 'young', 'experienced', 'senior'])
   
   # 2.4 Academic performance features
   # Normalize aggregate by qualification type
   if is_train:
       global qual_aggregate_stats
       qual_aggregate_stats = data.groupby('Qualification')['Aggregate'].agg(['mean', 'std'])
   
   data = data.merge(qual_aggregate_stats, left_on='Qualification', right_index=True, how='left')
   data['aggregate_normalized'] = (data['Aggregate'] - data['mean']) / data['std'].fillna(1)
   data['aggregate_above_avg'] = (data['aggregate_normalized'] > 0).astype(int)
   
   # Performance bins
   data['aggregate_bin'] = pd.cut(data['Aggregate'], 
                                  bins=[0, 50, 60, 70, 80, 100],
                                  labels=['fail', 'pass', 'good', 'very_good', 'excellent'])
   
   # 2.5 Institution-based features
   if is_train:
       global institution_stats
       institution_stats = data.groupby('Institution').agg({
           'Progress': ['mean', 'count'],
           'Successful': 'mean'
       })
       institution_stats.columns = ['inst_avg_progress', 'inst_count', 'inst_success_rate']
   
   data = data.merge(institution_stats, left_on='Institution', right_index=True, how='left')
   
   # 2.6 Demographic combinations
   data['gender_race'] = data['Gender'] + '_' + data['Race']
   data['young_female'] = ((data['Age'] < 25) & (data['Gender'] == 'Female')).astype(int)
   
   # 2.7 Interaction features
   data['age_aggregate_interaction'] = data['Age'] * data['Aggregate'] / 100
   data['matric_aggregate'] = data['is_matric'] * data['Aggregate']
   
   # Drop temporary columns
   data = data.drop(['mean', 'std'], axis=1)
   
   return data

# Apply preprocessing
print("Preprocessing training data...")
train_processed = preprocess_data(train_data, is_train=True)
print("Preprocessing test data...")
test_processed = preprocess_data(test_data, is_train=False)

# 3. ENCODE CATEGORICAL VARIABLES
print("\n3. ENCODING CATEGORICAL VARIABLES...")
print("="*50)

# Prepare encoders
categorical_features = ['Gender', 'Race', 'Institution', 'Qualification', 
                      'Industry', 'Company', 'age_group', 'aggregate_bin', 'gender_race']

# Store encoders globally for test set
encoders = {}

def encode_features(train_df, test_df):
   """Encode categorical features with proper handling of train/test"""
   train_encoded = train_df.copy()
   test_encoded = test_df.copy()
   
   for col in categorical_features:
       if col in train_encoded.columns:
           # Use LabelEncoder
           le = LabelEncoder()
           
           # Fit on training data
           train_encoded[col + '_encoded'] = le.fit_transform(train_encoded[col])
           
           # Transform test data with handling for unseen categories
           test_categories = test_encoded[col].unique()
           train_categories = train_encoded[col].unique()
           unseen = set(test_categories) - set(train_categories)
           
           if unseen:
               # Map unseen categories to most frequent training category
               most_frequent = train_encoded[col].mode()[0]
               test_encoded[col] = test_encoded[col].replace(list(unseen), most_frequent)
           
           test_encoded[col + '_encoded'] = le.transform(test_encoded[col])
           
           encoders[col] = le
   
   return train_encoded, test_encoded

train_encoded, test_encoded = encode_features(train_processed, test_processed)

# 4. HANDLE DISCIPLINES WITH MULTI-HOT ENCODING
print("\n4. MULTI-HOT ENCODING DISCIPLINES...")
print("="*50)

# Get all disciplines from training data
all_disciplines = [d for sublist in train_processed['Disciplines_list'] for d in sublist]
discipline_counts = Counter(all_disciplines)

# Select top disciplines
TOP_N_DISCIPLINES = 20
top_disciplines = set([d for d, _ in discipline_counts.most_common(TOP_N_DISCIPLINES)])

# Filter disciplines
def filter_disciplines(d_list):
   filtered = [d for d in d_list if d in top_disciplines]
   if not filtered:
       return ['Other']
   return filtered

train_encoded['Disciplines_filtered'] = train_encoded['Disciplines_list'].apply(filter_disciplines)
test_encoded['Disciplines_filtered'] = test_encoded['Disciplines_list'].apply(filter_disciplines)

# Multi-hot encode
mlb = MultiLabelBinarizer()
discipline_train = pd.DataFrame(
   mlb.fit_transform(train_encoded['Disciplines_filtered']),
   columns=[f"Disc_{d}" for d in mlb.classes_],
   index=train_encoded.index
)

discipline_test = pd.DataFrame(
   mlb.transform(test_encoded['Disciplines_filtered']),
   columns=[f"Disc_{d}" for d in mlb.classes_],
   index=test_encoded.index
)

# Merge back
train_final = pd.concat([train_encoded, discipline_train], axis=1)
test_final = pd.concat([test_encoded, discipline_test], axis=1)

# 5. SELECT FEATURES FOR MODELING
print("\n5. PREPARING FINAL FEATURE SET...")
print("="*50)

# Define feature columns
feature_cols = ['Age', 'Aggregate', 'num_disciplines', 'has_disciplines', 'is_matric', 
               'has_degree', 'aggregate_normalized', 'aggregate_above_avg',
               'inst_avg_progress', 'inst_count', 'inst_success_rate',
               'young_female', 'age_aggregate_interaction', 'matric_aggregate']

# Add encoded categorical features
feature_cols += [col + '_encoded' for col in categorical_features if col + '_encoded' in train_final.columns]

# Add discipline features
feature_cols += [col for col in train_final.columns if col.startswith('Disc_')]

# Prepare final datasets
X_train_full = train_final[feature_cols]
y_train_full = train_final['Progress']
X_test = test_final[feature_cols]

# Handle any missing values
imputer = SimpleImputer(strategy='median')
X_train_full = pd.DataFrame(imputer.fit_transform(X_train_full), columns=X_train_full.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print(f"Final feature count: {len(feature_cols)}")
print(f"Training shape: {X_train_full.shape}")
print(f"Test shape: {X_test.shape}")

# 6. MODEL DEVELOPMENT
print("\n6. MODEL TRAINING...")
print("="*50)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
   X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# 6.1 Train multiple models
models = {
   'rf': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, 
                               min_samples_leaf=2, random_state=42, n_jobs=-1),
   'gbm': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                                    min_samples_split=5, random_state=42),
   'xgb': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, 
                           min_child_weight=3, random_state=42, n_jobs=-1),
   'ridge': Ridge(alpha=10.0, random_state=42)
}

# Train and evaluate each model
model_scores = {}
for name, model in models.items():
   print(f"\nTraining {name}...")
   
   # Use scaled data for linear models
   if name in ['ridge', 'lasso']:
       model.fit(X_train_scaled, y_train)
       val_pred = model.predict(X_val_scaled)
   else:
       model.fit(X_train, y_train)
       val_pred = model.predict(X_val)
   
   # Clip predictions to valid range
   val_pred = np.clip(val_pred, 1, 5)
   
   # Calculate metrics
   mae = mean_absolute_error(y_val, val_pred)
   rmse = np.sqrt(mean_squared_error(y_val, val_pred))
   r2 = r2_score(y_val, val_pred)
   
   model_scores[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
   print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 6.2 Create ensemble model
print("\n\nCreating ensemble model...")
ensemble = VotingRegressor([
   ('rf', models['rf']),
   ('gbm', models['gbm']),
   ('xgb', models['xgb'])
])

# Train ensemble on full training data
ensemble.fit(X_train_full, y_train_full)

# 7. FEATURE IMPORTANCE ANALYSIS
print("\n7. FEATURE IMPORTANCE...")
print("="*50)

# Get feature importance from Random Forest
rf_model = models['rf']
rf_model.fit(X_train_full, y_train_full)
feature_importance = pd.DataFrame({
   'feature': X_train_full.columns,
   'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# 8. CROSS-VALIDATION
print("\n8. CROSS-VALIDATION...")
print("="*50)

# Use stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate ensemble
cv_scores = cross_val_score(ensemble, X_train_full, y_train_full, 
                          cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"Cross-validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 9. FINAL PREDICTIONS
print("\n9. GENERATING PREDICTIONS...")
print("="*50)

# Make predictions with ensemble
test_predictions = ensemble.predict(X_test)

# Post-process predictions
# 1. Clip to valid range
test_predictions = np.clip(test_predictions, 1, 5)

# 2. Round to nearest 0.5 for more realistic progress values
test_predictions = np.round(test_predictions * 2) / 2

# 10. CREATE SUBMISSION
print("\n10. CREATING SUBMISSION FILE...")
print("="*50)

submission = pd.DataFrame({
   'ID': test_data['ID'],
   'Progress': test_predictions
})

print("\nSubmission preview:")
print(submission.head(10))
print(f"\nPrediction distribution:")
print(submission['Progress'].value_counts().sort_index())

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")

# 11. ANALYSIS INSIGHTS FOR REPORT
print("\n11. KEY INSIGHTS FOR REPORT...")
print("="*50)

# Success factors analysis
print("\n1. Academic Performance Impact:")
progress_by_aggregate = train_data.groupby(pd.cut(train_data['Aggregate'], bins=5))['Progress'].mean()
print(progress_by_aggregate)

print("\n2. Qualification Type Impact:")
qual_progress = train_data.groupby('Qualification')['Progress'].agg(['mean', 'count'])
print(qual_progress.sort_values('mean', ascending=False).head(10))

print("\n3. Gender and Race Analysis:")
gender_race_progress = train_data.groupby(['Gender', 'Race'])['Progress'].mean().unstack()
print(gender_race_progress)

print("\n4. Age Impact:")
age_progress = train_data.groupby(pd.cut(train_data['Age'], bins=[15, 20, 25, 30, 35, 40]))['Progress'].mean()
print(age_progress)

print("\n5. Institution Performance:")
inst_performance = train_data.groupby('Institution').agg({
   'Progress': ['mean', 'count'],
   'Successful': 'mean'
}).sort_values(('Progress', 'mean'), ascending=False).head(10)
print(inst_performance)

print("\n" + "="*50)
print("COMPLETE! Model trained and predictions generated.")
print("="*50)

