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


# Install FLAML if not already installed
!pip install flaml -q

import numpy as np
import pandas as pd
import os
import warnings
import gc
from tqdm import tqdm
from itertools import combinations

# ML Libraries
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

# Gradient Boosting Libraries
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRegressor

# AutoML
from flaml import AutoML

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Check available files
print("Available files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Try to load original data if available
try:
    original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
    train = pd.concat([train, original], axis=0, ignore_index=True)
    print("âœ… Successfully loaded and combined with original dataset")
except FileNotFoundError:
    print("âš ï¸� Original dataset not found, proceeding with playground data only")

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Fix temperature column name
def rename_temperature_column(df):
    if 'Temparature' in df.columns:
        df = df.rename(columns={'Temparature': 'Temperature'})
        print("âœ… Column name corrected from 'Temparature' to 'Temperature'")
    return df
    
train = rename_temperature_column(train)
test = rename_temperature_column(test)

# Display basic info
print(f"\nTarget distribution:")
print(train['Fertilizer Name'].value_counts())

# Store original categorical columns before encoding
cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns 
            if col != "Fertilizer Name"]
print(f"\nCategorical columns: {cat_cols}")

# Label encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Encode target variable
fer_label_enc = LabelEncoder()
train["Fertilizer Name"] = fer_label_enc.fit_transform(train["Fertilizer Name"])
print(f"\nNumber of classes: {len(fer_label_enc.classes_)}")
print(f"Classes: {fer_label_enc.classes_}")

# Feature Engineering Function
def create_features(df, is_train=True):
    """Create domain-specific features for fertilizer prediction"""
    df_feat = df.copy()
    
    # NPK ratios and interactions
    df_feat['N_K_ratio'] = df_feat['Nitrogen'] / (df_feat['Potassium'] + 1)
    df_feat['N_P_ratio'] = df_feat['Nitrogen'] / (df_feat['Phosphorous'] + 1)
    df_feat['P_K_ratio'] = df_feat['Phosphorous'] / (df_feat['Potassium'] + 1)
    df_feat['NPK_total'] = df_feat['Nitrogen'] + df_feat['Phosphorous'] + df_feat['Potassium']
    
    # NPK balance metrics
    df_feat['NPK_std'] = df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
    df_feat['NPK_max_min_diff'] = (df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].max(axis=1) - 
                                   df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].min(axis=1))
    
    # Climate features
    df_feat['Heat_Humidity_Index'] = df_feat['Temperature'] * df_feat['Humidity'] / 100
    df_feat['Moisture_Temp_Interaction'] = df_feat['Moisture'] * df_feat['Temperature']
    df_feat['Climate_Stress'] = (df_feat['Temperature'] - 30)**2 + (df_feat['Humidity'] - 60)**2
    
    # Soil moisture efficiency
    df_feat['Moisture_Humidity_Ratio'] = df_feat['Moisture'] / (df_feat['Humidity'] + 1)
    
    # Crop-Soil interaction
    df_feat['Crop_Soil_Interaction'] = df_feat['Crop Type'] * 10 + df_feat['Soil Type']
    
    # Polynomial features for NPK
    df_feat['N_squared'] = df_feat['Nitrogen'] ** 2
    df_feat['P_squared'] = df_feat['Phosphorous'] ** 2
    df_feat['K_squared'] = df_feat['Potassium'] ** 2
    
    # Log transformations
    df_feat['log_Nitrogen'] = np.log1p(df_feat['Nitrogen'])
    df_feat['log_Phosphorous'] = np.log1p(df_feat['Phosphorous'])
    df_feat['log_Potassium'] = np.log1p(df_feat['Potassium'])
    
    # Additional domain features
    df_feat['NPK_N_dominance'] = df_feat['Nitrogen'] / (df_feat['NPK_total'] + 1)
    df_feat['NPK_P_dominance'] = df_feat['Phosphorous'] / (df_feat['NPK_total'] + 1)
    df_feat['NPK_K_dominance'] = df_feat['Potassium'] / (df_feat['NPK_total'] + 1)
    
    # Extreme weather indicators
    df_feat['High_temp'] = (df_feat['Temperature'] > 35).astype(int)
    df_feat['Low_moisture'] = (df_feat['Moisture'] < 30).astype(int)
    df_feat['High_humidity'] = (df_feat['Humidity'] > 65).astype(int)
    
    return df_feat

# MAP@3 evaluation metric
def mapk(actual, predicted, k=3):
    """Calculate Mean Average Precision @ k"""
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Prepare datasets
print("\n" + "="*60)
print("PREPARING FEATURES")
print("="*60)

X_base = train.drop(columns=["id", "Fertilizer Name"])
X_feat = create_features(train.drop(columns=["id", "Fertilizer Name"]))
y = train["Fertilizer Name"]

X_test_base = test.drop(columns=["id"])
X_test_feat = create_features(test.drop(columns=["id"]), is_train=False)

print(f"Base features: {X_base.shape[1]}")
print(f"Engineered features: {X_feat.shape[1]}")

# Convert categorical columns for XGBoost
for col in cat_cols:
    X_base[col] = X_base[col].astype("category")
    X_test_base[col] = X_test_base[col].astype("category")
    X_feat[col] = X_feat[col].astype("category")
    X_test_feat[col] = X_test_feat[col].astype("category")

# Initialize cross-validation
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize storage for predictions
n_classes = y.nunique()
n_train = len(train)
n_test = len(test)

# Out-of-fold predictions for meta-learning
oof_model1 = np.zeros((n_train, n_classes))  # Original XGBoost
oof_model2 = np.zeros((n_train, n_classes))  # LightGBM with features
oof_model3 = np.zeros((n_train, n_classes))  # FLAML AutoML

# Test predictions
test_pred_model1 = np.zeros((n_test, n_classes))
test_pred_model2 = np.zeros((n_test, n_classes))
test_pred_model3 = np.zeros((n_test, n_classes))

# Track scores
scores_model1 = []
scores_model2 = []
scores_model3 = []

print("\n" + "="*60)
print("MODEL 1: Original XGBoost")
print("="*60)

# Model 1: Original XGBoost
xgb_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',
    device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_base, y)):
    print(f'\nFold {fold + 1}/{FOLDS}')
    
    X_train, X_valid = X_base.iloc[train_idx], X_base.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    
    oof_model1[valid_idx] = xgb_model.predict_proba(X_valid)
    test_pred_model1 += xgb_model.predict_proba(X_test_base) / FOLDS
    
    # Evaluate
    top_3_preds = np.argsort(oof_model1[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    scores_model1.append(map3_score)
    print(f"âœ… Model 1 - Fold {fold + 1}: MAP@3 = {map3_score:.5f}")

print(f"\nModel 1 Average CV Score: {np.mean(scores_model1):.5f} (+/- {np.std(scores_model1):.5f})")

print("\n" + "="*60)
print("MODEL 2: LightGBM with Feature Engineering")
print("="*60)

# Model 2: LightGBM with engineered features
lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=8,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    objective='multiclass',
    num_class=n_classes,
    device='gpu' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    random_state=42,
    verbose=-1
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_feat, y)):
    print(f'\nFold {fold + 1}/{FOLDS}')
    
    X_train, X_valid = X_feat.iloc[train_idx], X_feat.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    oof_model2[valid_idx] = lgb_model.predict_proba(X_valid)
    test_pred_model2 += lgb_model.predict_proba(X_test_feat) / FOLDS
    
    # Evaluate
    top_3_preds = np.argsort(oof_model2[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    scores_model2.append(map3_score)
    print(f"âœ… Model 2 - Fold {fold + 1}: MAP@3 = {map3_score:.5f}")

print(f"\nModel 2 Average CV Score: {np.mean(scores_model2):.5f} (+/- {np.std(scores_model2):.5f})")

print("\n" + "="*60)
print("MODEL 3: FLAML AutoML")
print("="*60)

# Model 3: FLAML AutoML
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_feat, y)):
    print(f'\nFold {fold + 1}/{FOLDS}')
    
    X_train, X_valid = X_feat.iloc[train_idx], X_feat.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Configure FLAML
    automl = AutoML()
    automl_settings = {
        "time_budget": 180,  # 3 minutes per fold (reduced for faster execution)
        "metric": 'accuracy',
        "task": 'classification',
        "n_jobs": -1,
        "estimator_list": ['lgbm', 'xgboost', 'catboost'],
        "seed": 42,
        "early_stop": True,
        "eval_method": "cv",
        "n_splits": 3,
        "verbose": 0
    }
    
    try:
        automl.fit(X_train, y_train, **automl_settings)
        
        oof_model3[valid_idx] = automl.predict_proba(X_valid)
        test_pred_model3 += automl.predict_proba(X_test_feat) / FOLDS
        
        # Evaluate
        top_3_preds = np.argsort(oof_model3[valid_idx], axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_valid]
        map3_score = mapk(actual, top_3_preds)
        scores_model3.append(map3_score)
        print(f"âœ… Model 3 - Fold {fold + 1}: MAP@3 = {map3_score:.5f}")
        print(f"   Best model: {automl.best_estimator}")
    except Exception as e:
        print(f"âš ï¸� FLAML error in fold {fold + 1}: {str(e)}")
        # Fallback to CatBoost if FLAML fails
        print("   Using CatBoost as fallback...")
        cb_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function='MultiClass',
            verbose=False,
            random_state=42
        )
        cb_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
        oof_model3[valid_idx] = cb_model.predict_proba(X_valid)
        test_pred_model3 += cb_model.predict_proba(X_test_feat) / FOLDS
        
        top_3_preds = np.argsort(oof_model3[valid_idx], axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_valid]
        map3_score = mapk(actual, top_3_preds)
        scores_model3.append(map3_score)
        print(f"âœ… Model 3 - Fold {fold + 1}: MAP@3 = {map3_score:.5f}")

print(f"\nModel 3 Average CV Score: {np.mean(scores_model3):.5f} (+/- {np.std(scores_model3):.5f})")

print("\n" + "="*60)
print("META-LEARNING: Stacking Ensemble")
print("="*60)

# Prepare meta-features
def create_meta_features(probs1, probs2, probs3):
    """Create meta-features from base model predictions"""
    meta_features = []
    
    # Raw probabilities
    meta_features.append(probs1)
    meta_features.append(probs2)
    meta_features.append(probs3)
    
    # Max probabilities
    meta_features.append(np.max([probs1, probs2, probs3], axis=0))
    
    # Mean probabilities
    meta_features.append(np.mean([probs1, probs2, probs3], axis=0))
    
    # Standard deviation
    meta_features.append(np.std([probs1, probs2, probs3], axis=0))
    
    # Entropy of each model's predictions
    eps = 1e-15
    entropy1 = -np.sum(probs1 * np.log(probs1 + eps), axis=1, keepdims=True)
    entropy2 = -np.sum(probs2 * np.log(probs2 + eps), axis=1, keepdims=True)
    entropy3 = -np.sum(probs3 * np.log(probs3 + eps), axis=1, keepdims=True)
    
    meta_features.extend([entropy1, entropy2, entropy3])
    
    # Top-1 confidence for each model
    conf1 = np.max(probs1, axis=1, keepdims=True)
    conf2 = np.max(probs2, axis=1, keepdims=True)
    conf3 = np.max(probs3, axis=1, keepdims=True)
    
    meta_features.extend([conf1, conf2, conf3])
    
    return np.hstack(meta_features)

# Create meta-training data
meta_train = create_meta_features(oof_model1, oof_model2, oof_model3)
meta_test = create_meta_features(test_pred_model1, test_pred_model2, test_pred_model3)

print(f"Meta features shape: {meta_train.shape}")

# Train meta-learner using XGBoost
meta_oof = np.zeros((n_train, n_classes))
meta_test_pred = np.zeros((n_test, n_classes))
scores_meta = []

meta_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    random_state=42,
    tree_method='hist',
    device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(meta_train, y)):
    print(f'\nMeta-learner Fold {fold + 1}/{FOLDS}')
    
    X_train_meta = meta_train[train_idx]
    X_valid_meta = meta_train[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    meta_model.fit(
        X_train_meta, y_train,
        eval_set=[(X_valid_meta, y_valid)],
        early_stopping_rounds=50,
        verbose=0
    )
    
    meta_oof[valid_idx] = meta_model.predict_proba(X_valid_meta)
    meta_test_pred += meta_model.predict_proba(meta_test) / FOLDS
    
    # Evaluate
    top_3_preds = np.argsort(meta_oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    scores_meta.append(map3_score)
    print(f"âœ… Meta-learner Fold {fold + 1}: MAP@3 = {map3_score:.5f}")

print(f"\nMeta-learner Average CV Score: {np.mean(scores_meta):.5f} (+/- {np.std(scores_meta):.5f})")

print("\n" + "="*60)
print("FINAL ENSEMBLE RESULTS")
print("="*60)

# Compare all approaches
approaches = [
    ("Model 1 (Original XGB)", oof_model1),
    ("Model 2 (LGB + Features)", oof_model2),
    ("Model 3 (AutoML/CatBoost)", oof_model3),
    ("Simple Average", (oof_model1 + oof_model2 + oof_model3) / 3),
    ("Weighted Average", 0.3 * oof_model1 + 0.4 * oof_model2 + 0.3 * oof_model3),
    ("Meta-learning Stacking", meta_oof)
]

best_score = -1
best_approach = None
best_predictions = None

for name, predictions in approaches:
    top_3_preds = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y]
    map3_score = mapk(actual, top_3_preds)
    print(f"{name}: MAP@3 = {map3_score:.5f}")
    
    if map3_score > best_score:
        best_score = map3_score
        best_approach = name
        best_predictions = predictions if name != "Meta-learning Stacking" else meta_test_pred

print(f"\nğŸ�† Best approach: {best_approach} with MAP@3 = {best_score:.5f}")

# Generate final submission using best approach
print("\n" + "="*60)
print("GENERATING SUBMISSION")
print("="*60)

# Use meta-learning predictions (usually the best)
final_test_predictions = meta_test_pred

top_3_preds = np.argsort(final_test_predictions, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv('submission.csv', index=False)
print(f"âœ… Submission file saved as 'submission.csv'")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 5 predictions:")
print(submission.head())

# Feature importance from Model 2
print("\n" + "="*60)
print("TOP 20 FEATURE IMPORTANCES (Model 2)")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_feat.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# Save individual model predictions for post-processing
try:
    np.save('oof_model1.npy', oof_model1)
    np.save('oof_model2.npy', oof_model2)
    np.save('oof_model3.npy', oof_model3)
    np.save('meta_oof.npy', meta_oof)
    np.save('test_pred_model1.npy', test_pred_model1)
    np.save('test_pred_model2.npy', test_pred_model2)
    np.save('test_pred_model3.npy', test_pred_model3)
    np.save('meta_test_pred.npy', meta_test_pred)
    print("\nâœ… All predictions saved for future analysis!")
except Exception as e:
    print(f"\nâš ï¸� Could not save predictions: {str(e)}")

# Optional: Visualize model performance
try:
    import matplotlib.pyplot as plt
    
    # Model comparison plot
    model_names = ['XGBoost\n(Original)', 'LightGBM\n(+Features)', 'AutoML', 
                   'Simple\nAverage', 'Weighted\nAverage', 'Meta-learning\nStacking']
    scores = []
    
    for _, predictions in approaches:
        top_3_preds = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y]
        scores.append(mapk(actual, top_3_preds))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.title('Model Performance Comparison (MAP@3)', fontsize=14, fontweight='bold')
    plt.ylabel('MAP@3 Score', fontsize=12)
    plt.ylim(min(scores) * 0.99, max(scores) * 1.01)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nâœ… Performance visualization saved as 'model_comparison.png'")
    
except ImportError:
    print("\nâš ï¸� Matplotlib not available for visualization")

print("\n" + "="*60)
print("ENSEMBLE INSIGHTS & RECOMMENDATIONS")
print("="*60)
print("âœ… Successfully trained 3 diverse models with meta-learning ensemble")
print("\nKey Insights:")
print("1. Feature Engineering: NPK ratios and climate interactions are crucial")
print("2. Model Diversity: Different algorithms capture different patterns")
print("3. Meta-learning: Learns optimal combination based on prediction confidence")
print("\nNext Steps for Improvement:")
print("- Add TF-IDF features from crop/soil combinations")
print("- Implement pseudo-labeling with high-confidence test predictions")
print("- Try TabNet or other neural architectures as additional base models")
print("- Add post-processing rules based on agronomic knowledge")
print("- Experiment with different meta-learner architectures")

print("\nğŸ�¯ Good luck with your submission!")

