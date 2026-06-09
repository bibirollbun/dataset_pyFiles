# -*- coding: utf-8 -*-
"""
Optimal Fertilizer Prediction - Grand Prize Solution
Competition: Playground Series Season 5 Episode 6
Author: [Your Name]
Last Updated: [Date]
"""

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

# Model and Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from lightgbm import early_stopping, log_evaluation
from bayes_opt import BayesianOptimization

# Feature Engineering Utilities
from sklearn.feature_selection import mutual_info_classif


# =============================================
# GLOBAL SETTINGS AND CONFIGURATIONS
# =============================================
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', 100)
np.random.seed(42)


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================
print("ğŸ“Š [1/6] Loading and preprocessing datasets...")

# Load competition data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Attempt to load external data
try:
    original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
    print("âœ… External dataset loaded successfully")
    train = pd.concat([train, original], axis=0).reset_index(drop=True)
    print(f"ğŸ”„ Combined dataset shape: {train.shape}")
except Exception as e:
    print(f"âš ï¸� External dataset not available. Using competition data only. Error: {e}")

# Fix column name spelling
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)


# =============================================
# ADVANCED FEATURE ENGINEERING
# =============================================
print("ğŸ”§ [2/6] Performing advanced feature engineering...")

def create_features(df):
    # Nutrient ratios with smoothing
    df['N/P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['N/K'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['P/K'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    # Nutrient aggregates
    df['Nutrient_Sum'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)
    df['Nutrient_Avg'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)
    
    # Nutrient imbalances
    df['N_Imbalance'] = (df['Nitrogen'] - df['Nutrient_Avg'])**2
    df['P_Imbalance'] = (df['Phosphorous'] - df['Nutrient_Avg'])**2
    df['K_Imbalance'] = (df['Potassium'] - df['Nutrient_Avg'])**2
    
    # Environmental interactions
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
    df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
    
    # Polynomial features
    df['Temp_squared'] = df['Temperature']**2
    df['Humidity_sqrt'] = np.sqrt(df['Humidity'])
    
    # Binning continuous features
    for col in ['Temperature', 'Humidity', 'Moisture']:
        df[f'{col}_bin'] = pd.cut(df[col], bins=5, labels=False)
    
    return df

train = create_features(train)
test = create_features(test)


# =============================================
# ENCODING AND FINAL PREPARATION
# =============================================
print("ğŸ”  [3/6] Encoding categorical features...")

# Categorical encoding
cat_cols = ['Soil Type', 'Crop Type']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = ordinal_encoder.fit_transform(train[cat_cols])
test[cat_cols] = ordinal_encoder.transform(test[cat_cols])

# Target encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# Prepare data
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop('id', axis=1)


# =============================================
# BAYESIAN OPTIMIZATION FOR HYPERPARAMETERS
# =============================================
print("âš™ï¸� [4/6] Optimizing hyperparameters with Bayesian Optimization...")

def xgb_optimization(max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'random_state': 42,
        'n_jobs': -1
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params, n_estimators=500)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=50,
                 verbose=0)
        
        val_pred = model.predict_proba(X_val)
        top3 = np.argsort(-val_pred, axis=1)[:, :3]
        score = np.mean([1 if y_val.iloc[i] in top3[i] else 0 for i in range(len(y_val))])
        cv_scores.append(score)
    
    return np.mean(cv_scores)

optimizer = BayesianOptimization(
    f=xgb_optimization,
    pbounds={
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    },
    random_state=42,
    verbose=0
)

optimizer.maximize(init_points=5, n_iter=10)
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])


# =============================================
# ENSEMBLE MODEL TRAINING
# =============================================
print("ğŸ¤– [5/6] Training optimized ensemble models...")

# Model configurations with optimized parameters
models = {
    'xgb': XGBClassifier(
        **best_params,
        n_estimators=2000,
        objective='multi:softprob',
        num_class=len(le.classes_),
        random_state=42,
        n_jobs=-1
    ),
    'lgbm': LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multiclass',
        num_class=len(le.classes_),
        random_state=42,
        n_jobs=-1
    ),
    'catboost': CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=1.0,
        random_strength=0.1,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
}

# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), len(le.classes_)))

# Custom MAP@3 metric
def map3(y_true, y_pred):
    def apk(actual, predicted, k=3):
        predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        seen = set()
        for i, p in enumerate(predicted):
            if p in actual and p not in seen:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
                seen.add(p)
        return score / min(len(actual), k)
    return np.mean([apk([a], p) for a, p in zip(y_true, y_pred)])

# Training loop with cross-validation
for model_name, model in models.items():
    print(f"\nğŸ”¥ Training {model_name.upper()} with {n_splits}-fold CV")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  ğŸŒ€ Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_name == 'xgb':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        elif model_name == 'lgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[
                    early_stopping(stopping_rounds=50),
                    log_evaluation(100)
                ]
            )
        elif model_name == 'catboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        
        # Generate predictions
        val_preds = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / (n_splits * len(models))
        
        # Calculate and store fold score
        top3 = np.argsort(-val_preds, axis=1)[:, :3]
        fold_score = map3(y_val, top3)
        fold_scores.append(fold_score)
        print(f"  ğŸ�¯ Fold {fold + 1} MAP@3: {fold_score:.5f}")
        
        # Memory management
        del X_train, X_val, y_train, y_val, val_preds
        gc.collect()
    
    # Print model summary
    print(f"\nâœ¨ {model_name.upper()} CV Results:")
    print(f"   Mean MAP@3: {np.mean(fold_scores):.5f} Â± {np.std(fold_scores):.5f}")


# =============================================
# GENERATE FINAL SUBMISSION
# =============================================
print("\nğŸ�¯ [6/6] Generating final submission...")

# Ensemble predictions
top3_preds = np.argsort(-test_preds, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

# Create submission file
submission['Fertilizer Name'] = [' '.join(row) for row in top3_labels]
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


# =============================================
# FEATURE IMPORTANCE VISUALIZATION
# =============================================
print("\nğŸ“Š Feature importance analysis...")

fig, ax = plt.subplots(figsize=(14, 10))
xgb_feat_imp = models['xgb'].feature_importances_
sorted_idx = np.argsort(xgb_feat_imp)
ax.barh(range(len(sorted_idx)), xgb_feat_imp[sorted_idx], align='center', color='#1f77b4')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=12)
ax.set_title('XGBoost Feature Importance', fontsize=16, pad=20)
ax.set_xlabel('Importance Score', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nğŸ�† All done! Ready for submission to win the competition!")

