import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
print(f"Train: {train.shape} | Test: {test.shape}")


def create_features(df):
    df = df.copy()
    
    # Duration features
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['LogDuration'] = np.log1p(df['TrackDurationMs'])
    
    # Key interactions
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Mood_Live'] = df['MoodScore'] * df['LivePerformanceLikelihood'] 
    df['Acoustic_Vocal'] = df['AcousticQuality'] * df['VocalContent']
    
    # Ratios (safe division)
    eps = 1e-6
    df['Energy_over_Rhythm'] = df['Energy'] / (df['RhythmScore'] + eps)
    df['Vocal_over_Acoustic'] = df['VocalContent'] / (df['AcousticQuality'] + eps)
    
    # Square features for top predictors
    for col in ['RhythmScore', 'Energy', 'MoodScore']:
        df[f'{col}_sq'] = df[col] ** 2
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print(f"Features created: {len(train_fe.columns) - 2} total features")

# Prepare data
TARGET = 'BeatsPerMinute'
ID = 'id'
features = [c for c in train_fe.columns if c not in [TARGET, ID]]

X = train_fe[features]
y = train_fe[TARGET]
X_test = test_fe[features]

print(f"Final shape: X={X.shape}, X_test={X_test.shape}")


# Simple but effective models
models = [
    {
        'name': 'LGBM_1',
        'model': LGBMRegressor(
            num_leaves=33, learning_rate=0.03, n_estimators=300,
            max_depth=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    },
    {
        'name': 'LGBM_2', 
        'model': LGBMRegressor(
            num_leaves=28, learning_rate=0.02, n_estimators=250,
            max_depth=12, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=0.5,
            random_state=52, n_jobs=-1, verbose=-1
        )
    }
]

# Training with CV
def train_ensemble(X, y, X_test, models, n_splits=3):
    test_preds = np.zeros((len(X_test), len(models)))
    cv_scores = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for idx, model_config in enumerate(models):
        model = model_config['model']
        name = model_config['name']
        
        print(f"ğŸš€ Training {name}...")
        
        # Cross-validation
        cv_rmse = np.sqrt(-cross_val_score(
            model, X, y, scoring='neg_mean_squared_error', cv=kf, n_jobs=1
        ))
        
        print(f"   CV RMSE: {cv_rmse.mean():.4f} (Â±{cv_rmse.std():.4f})")
        cv_scores.append(cv_rmse.mean())
        
        # Train and predict
        model.fit(X, y)
        test_preds[:, idx] = model.predict(X_test)
        
        # Training RMSE
        train_pred = model.predict(X)
        train_rmse = mean_squared_error(y, train_pred, squared=False)
        print(f"   Train RMSE: {train_rmse:.4f}")
        print()
    
    return test_preds, cv_scores

# Train models
test_preds, cv_scores = train_ensemble(X, y, X_test, models)


# Weighted ensemble (better than simple average)
weights = 1 / np.array(cv_scores)
weights = weights / weights.sum()
final_preds = np.average(test_preds, axis=1, weights=weights)

print(f"ğŸ�¯ Final ensemble CV score: {np.mean(cv_scores):.4f}")
print(f"ğŸ“Š Model weights: {dict(zip([m['name'] for m in models], weights.round(3)))}")

# Create submission
submission = pd.DataFrame({
    ID: test[ID],
    TARGET: final_preds
})

submission.to_csv('submission.csv', index=False)
print(f"âœ… Submission saved! Shape: {submission.shape}")
print("\nğŸ“‹ Sample predictions:")
print(submission.head())




