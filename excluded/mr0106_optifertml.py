# -*- coding: utf-8 -*-
"""
Optimal Fertilizer Prediction - Final Corrected Solution
Competition: Playground Series Season 5 Episode 6
Author: [Your Name]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import gc
import warnings
warnings.filterwarnings('ignore')


# Set global styles
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', 100)



## Data Loading
print("ğŸ“Š Loading datasets...")
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# Try to load original data if available
try:
    original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
    print("âœ… Original dataset loaded successfully")
    print("ğŸ”„ Combining datasets...")
    train = pd.concat([train, original], axis=0).reset_index(drop=True)
except:
    print("âš ï¸� Original dataset not found, using only competition data")


## Feature Engineering
print("ğŸ”§ Feature engineering...")


# Correct spelling of 'Temperature'
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)


# Nutrient ratios and interactions
for df in [train, test]:
    df['N/P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['N/K'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['P/K'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Nutrient_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['Nutrient_Imbalance'] = ((df['Nitrogen'] - df['Nutrient_Sum']/3)**2 + 
                               (df['Phosphorous'] - df['Nutrient_Sum']/3)**2 + 
                               (df['Potassium'] - df['Nutrient_Sum']/3)**2)
    
    # Environmental interactions
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
    df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
    
    # Binning continuous features
    df['Temp_bin'] = pd.cut(df['Temperature'], bins=5, labels=False)
    df['Humidity_bin'] = pd.cut(df['Humidity'], bins=5, labels=False)
    df['Moisture_bin'] = pd.cut(df['Moisture'], bins=5, labels=False)


# Encode categorical features
print("ğŸ”  Encoding categorical features...")
cat_cols = ['Soil Type', 'Crop Type']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = ordinal_encoder.fit_transform(train[cat_cols])
test[cat_cols] = ordinal_encoder.transform(test[cat_cols])


# Target encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])


## Model Training
print("ğŸ¤– Training models...")


# Prepare data
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop('id', axis=1)


# Custom MAP@3 metric
def map3(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    def apk(actual, predicted, k=3):
        if len(predicted) > k:
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


# Model configurations
models = {
    'xgb': XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ),
    'lgbm': LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ),
    'catboost': CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
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
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), len(le.classes_)))


# -*- coding: utf-8 -*-
from lightgbm import early_stopping, log_evaluation

# Training loop - Fully corrected version
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{n_splits}")
        
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
                    log_evaluation(100)  # Prints progress every 100 iterations
                ]
            )
            
        elif model_name == 'catboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        
        # Test predictions
        test_preds += model.predict_proba(X_test) / (n_splits * len(models))
        
        # Calculate MAP@3 for this fold
        val_preds = model.predict_proba(X_val)
        top3 = np.argsort(-val_preds, axis=1)[:, :3]
        fold_score = map3(y_val, top3)
        fold_scores.append(fold_score)
        print(f"  Fold {fold + 1} MAP@3: {fold_score:.5f}")
        
        # Clear memory
        del X_train, X_val, y_train, y_val, val_preds
        gc.collect()
    
    print(f"\n{model_name} CV MAP@3: {np.mean(fold_scores):.5f} Â± {np.std(fold_scores):.5f}")


# Generate final predictions
print("\nğŸ�¯ Generating submission...")
top3_preds = np.argsort(-test_preds, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

submission['Fertilizer Name'] = [' '.join(row) for row in top3_labels]
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


# Feature importance
print("\nğŸ“ˆ Feature importance analysis...")
fig, ax = plt.subplots(figsize=(12, 8))
xgb_feat_imp = models['xgb'].feature_importances_
sorted_idx = np.argsort(xgb_feat_imp)
ax.barh(range(len(sorted_idx)), xgb_feat_imp[sorted_idx], align='center')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(X.columns)[sorted_idx])
ax.set_title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

print("\nğŸ‘� All done! Good luck with the competition!")

