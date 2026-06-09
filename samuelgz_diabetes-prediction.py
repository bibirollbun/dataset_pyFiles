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







import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime


start_time = time.time()
print("=" * 80)
print(" COMPLETE 8-MODEL EVALUATION PIPELINE")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 80)

print("\n1. LOADING DATA...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
print(" Loaded from Kaggle")

# Use 150K samples for evaluation
SAMPLE_SIZE = 150000
if len(train_df) > SAMPLE_SIZE:
    train_df = train_df.sample(SAMPLE_SIZE, random_state=42)
    print(f" Using {SAMPLE_SIZE:,} samples for evaluation")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("2. DATA ANALYSIS")
print("=" * 80)

print("Target distribution:")
diabetes_rate = train_df['diagnosed_diabetes'].mean()
print(f"  Diabetes: {diabetes_rate:.2%}")
print(f"  Non-diabetes: {1-diabetes_rate:.2%}")
print(f"  Ratio: {(1-diabetes_rate)/diabetes_rate:.1f}:1")

print("\nFeature types:")
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in num_cols if c not in ['id', 'diagnosed_diabetes']]

print(f"  Numerical: {len(num_cols)}")
print(f"  Categorical: {len(cat_cols)}")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("3. FEATURE ENGINEERING")
print("=" * 80)

def create_features(df):
    df = df.copy()
    # Health risk features
    df['bmi_age'] = df['bmi'] * df['age'] / 100
    df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    
    # Blood pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Cholesterol ratios
    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
    # Lifestyle score
    df['health_score'] = (
        df['diet_score'] * 0.3 +
        np.log1p(df['physical_activity_minutes_per_week']) * 0.3 +
        (8 - df['sleep_hours_per_day']).clip(0, 4) * 0.2 +
        (6 - df['screen_time_hours_per_day']).clip(0, 4) * 0.2
    )
    
    # Risk flags
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)
print(f"Added 9 engineered features")
print(f"Total features: {train_df.shape[1] - 2}")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("4. DATA PREPARATION")
print("=" * 80)

X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

# Update column lists
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Training features: {X.shape}")
print(f"Test features: {X_test.shape}")
print(f"Target: {y.shape}")
print(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("5. PREPROCESSING PIPELINE")
print("=" * 80)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simple preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

print("Preprocessor configured")
print(f"Will process {len(num_cols)} numerical features")
print(f"Will process {len(cat_cols)} categorical features")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("6. 8-MODEL EVALUATION")
print("=" * 80)


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


# Define 5 different models
models = {
    'LogisticRegression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced',
        C=0.1, solver='liblinear', n_jobs=1
    ),
    
    'RandomForest': RandomForestClassifier(
        random_state=42, n_estimators=150, class_weight='balanced',
        max_depth=12, min_samples_split=10, n_jobs=-1
    ),
    
    'ExtraTrees': ExtraTreesClassifier(
        random_state=42, n_estimators=150, class_weight='balanced',
        max_depth=12, min_samples_split=10, n_jobs=-1
    ),
    
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42, n_estimators=150,
        learning_rate=0.05, max_depth=6, subsample=0.8
    ),
    
    'XGBoost': XGBClassifier(
        random_state=42, n_estimators=150,
        learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', use_label_encoder=False,
        verbosity=0, n_jobs=-1
    ),
    
    'LightGBM': LGBMClassifier(
        random_state=42, n_estimators=150,
        learning_rate=0.05, num_leaves=31, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        verbosity=-1, n_jobs=-1
    ),
    
    'KNeighbors': KNeighborsClassifier(
        n_neighbors=50, weights='distance',
        n_jobs=-1
    ),
    
    'SGDClassifier': SGDClassifier(
        random_state=42, max_iter=1000,
        loss='log_loss', class_weight='balanced',
        n_jobs=-1, early_stopping=True
    )
}



print("Evaluating 8 models with 3-fold cross-validation...")
print("-" * 60)

best_score = 0
best_model = None
best_model_name = ""
results = {}

for i, (name, model) in enumerate(models.items(), 1):
    model_start = time.time()
    
    print(f"\n[{i}/8] {name}...")
    
    try:
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # 3-fold CV
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, 
                               scoring='roc_auc', n_jobs=-1)
        
        elapsed = time.time() - model_start
        
        results[name] = {
            'auc': scores.mean(),
            'std': scores.std(),
            'time': elapsed,
            'scores': scores
        }
        
        print(f"  AUC: {scores.mean():.6f} (±{scores.std():.6f})")
        print(f"  Time: {elapsed:.1f}s")
        
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
            best_model_name = name
            print(f"  New best!")
            
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print(f"Best AUC: {best_score:.6f}")
print("=" * 60)
print(f"Total evaluation time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("7. RESULTS ANALYSIS")
print("=" * 80)

# Create ranking
ranking = []
for name, res in results.items():
    ranking.append({
        'Model': name,
        'AUC': f"{res['auc']:.6f}",
        'Std': f"{res['std']:.6f}",
        'Time_s': f"{res['time']:.1f}",
        'Diff': f"{res['auc'] - best_score:+.6f}"
    })

ranking_df = pd.DataFrame(ranking)
ranking_df = ranking_df.sort_values('AUC', ascending=False)

print("\nMODEL RANKING:")
print(ranking_df.to_string(index=False))


# Insights
print(f"\nINSIGHTS:")
print(f"1. Best model: {best_model_name} (AUC: {best_score:.6f})")
print(f"2. Number of models >0.71 AUC: {sum(float(r['AUC']) > 0.71 for r in ranking)}")
print(f"3. Fastest model: {ranking_df.iloc[ranking_df['Time_s'].astype(float).argmin()]['Model']}")
print(f"4. Total evaluation time: {sum(float(r['Time_s']) for r in ranking):.1f}s")

print(f"\nTime: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("8. LOADING FULL DATASET")
print("=" * 80)

print("Loading complete dataset for final training...")

full_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
full_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Apply features
full_train = create_features(full_train)
full_test = create_features(full_test)

X_full = full_train.drop(['id', 'diagnosed_diabetes'], axis=1)
y_full = full_train['diagnosed_diabetes']
X_test_full = full_test.drop('id', axis=1)

print(f"Full dataset loaded")
print(f"Training: {X_full.shape}, Test: {X_test_full.shape}")
print(f"Time: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print(f"9. TRAINING {best_model_name} ON FULL DATA")
print("=" * 80)

# Create final pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

print(f"Training on {len(X_full):,} samples...")
train_start = time.time()
final_pipeline.fit(X_full, y_full)
train_time = time.time() - train_start

print(f"Training completed in {train_time:.1f}s")
print(f"Total time: {time.time()-start_time:.1f}s")





# print("\n" + "=" * 80)
# print("SAVING MODEL...")
# print("=" * 80)

# import joblib
# import json
# import os

# # Save the trained pipeline
# model_filename = f'diabetes_model_{best_model_name}.pkl'
# joblib.dump(final_pipeline, model_filename, compress=3)
# print(f"Model saved: {model_filename}")

# # Save feature names
# feature_names = list(X_full.columns)
# with open('feature_names.json', 'w') as f:
#     json.dump(feature_names, f)
# print(f"Feature names saved: feature_names.json")

# # Save model metadata
# metadata = {
#     'model_name': best_model_name,
#     'model_type': type(best_model).__name__,
#     'cv_score': best_score,
#     'validation_auc': None,  # Will update after validation
#     'training_samples': len(X_full),
#     'training_features': len(feature_names),
#     'training_time': train_time,
#     'features': feature_names,
#     'date_saved': time.strftime("%Y-%m-%d %H:%M:%S")
# }

# joblib.dump(metadata, 'model_metadata.pkl')
# print(f"Metadata saved: model_metadata.pkl")

# print(f"\nModel size: {os.path.getsize(model_filename)/1024:.1f} KB")
# print(f"Features used: {len(feature_names)}")
# print(f"Best model: {best_model_name} (AUC: {best_score:.6f})")

# print("\n" + "=" * 80)
# print("MODEL SAVED!")
# print("=" * 80)





print("\n" + "=" * 80)
print("10. VALIDATION CHECK")
print("=" * 80)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Quick validation
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_full, y_full, test_size=0.1, random_state=42, stratify=y_full
)

val_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

val_pipeline.fit(X_train_val, y_train_val)
val_preds = val_pipeline.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)

print(f"Validation AUC: {val_auc:.6f}")
print(f"CV AUC was: {best_score:.6f}")
print(f"Difference: {val_auc - best_score:+.6f}")

if abs(val_auc - best_score) < 0.005:
    print("Results consistent")
else:
    print("Some variation")

print(f"\nTime: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("11. MAKING PREDICTIONS")
print("=" * 80)

print(f"Predicting {len(X_test_full):,} test samples...")
pred_start = time.time()
test_predictions = final_pipeline.predict_proba(X_test_full)[:, 1]
pred_time = time.time() - pred_start

print(f"Predictions completed in {pred_time:.1f}s")

print(f"\nPrediction stats:")
print(f"  Mean: {test_predictions.mean():.6f}")
print(f"  Std:  {test_predictions.std():.6f}")
print(f"  Min:  {test_predictions.min():.6f}")
print(f"  Max:  {test_predictions.max():.6f}")
print(f"  <0.1: {np.mean(test_predictions < 0.1):.1%}")
print(f"  >0.9: {np.mean(test_predictions > 0.9):.1%}")

print(f"\nTime: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("12. CREATE SUBMISSION FILE")
print("=" * 80)

submission = pd.DataFrame({
    'id': full_test['id'],
    'diagnosed_diabetes': test_predictions
})

# Ensure valid probabilities
submission['diagnosed_diabetes'] = submission['diagnosed_diabetes'].clip(0.00001, 0.99999)

submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"Submission saved: {submission_file}")
print(f"Size: {submission.shape}")
print(f"Memory: {submission.memory_usage(deep=True).sum()/1024:.1f} KB")

print("\nSample predictions:")
print(submission.head().to_string(index=False))
print(f"\nTime: {time.time()-start_time:.1f}s")





print("\n" + "=" * 80)
print("13. FINAL SUMMARY")
print("=" * 80)

total_time = time.time() - start_time
minutes = total_time / 60

print(f"8-MODEL PIPELINE COMPLETED IN {minutes:.1f} MINUTES")
print("-" * 50)

print(f"\nBEST MODEL: {best_model_name}")
print(f"   CV AUC: {best_score:.6f}")
print(f"   Validation AUC: {val_auc:.6f}")

print(f"\nTOP 3 MODELS:")
sorted_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
for i, (name, res) in enumerate(sorted_models[:3], 1):
    star = " ★" if i == 1 else ""
    print(f"   {i}. {name:20s}: {res['auc']:.6f} ({res['time']:.1f}s){star}")

print(f"\nSUBMISSION:")
print(f"   File: {submission_file}")
print(f"   Predictions: {len(test_predictions):,}")
print(f"   Mean probability: {test_predictions.mean():.4f}")

print(f"\nTIMING:")
print(f"   Model evaluation: {sum(r['time'] for r in results.values()):.1f}s")
print(f"   Full training: {train_time:.1f}s")
print(f"   Predictions: {pred_time:.1f}s")
print(f"   Total: {total_time:.1f}s ({minutes:.1f} minutes)")

print(f"\nEXPECTED KAGGLE SCORE:")
expected_range = f"{best_score-0.005:.4f} - {best_score+0.005:.4f}"
print(f"   AUC likely between: {expected_range}")

print("\n" + "=" * 80)


print(f"\nTotal time: {minutes:.1f} minutes")
print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")





import os

print("CHECKING OUTPUT FILES:")
print("=" * 50)

files = os.listdir()
if not files:
    print("Folder is EMPTY!")
    print("This means the saving code didn't work properly.")
else:
    for file in files:
        path = f'{file}'
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"{file} ({size_kb:.1f} KB)")


# import os
# import shutil
# from IPython.display import HTML, display

# print(" CREATING DOWNLOAD LINKS...")
# print("=" * 60)

# # Create a simple HTML page with download links
# files = ['diabetes_model_LightGBM.pkl', 'feature_names.json', 'model_metadata.pkl', 'submission_file.csv']

# html_content = "<h3> Download Your Files:</h3><ul>"

# for file in files:
#     path = f'{file}'
#     if os.path.exists(path):
#         # Create a data URL for direct download
#         import base64
#         with open(path, 'rb') as f:
#             data = f.read()
#             b64 = base64.b64encode(data).decode()
        
#         html_content += f"""
#         <li>
#             <a href="data:application/octet-stream;base64,{b64}" download="{file}">
#                 {file} ({len(data)/1024:.1f} KB)
#             </a>
#         </li>
#         """
#     else:
#         html_content += f"<li> {file} not found</li>"

# html_content += "</ul>"
# display(HTML(html_content))

