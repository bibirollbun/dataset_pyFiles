%%time
!pip install autogluon==1.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from autogluon.tabular import TabularPredictor
import os
import sklearn
print("AutoGluon ready, sklearn version:", sklearn.__version__)


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


TARGET = 'diagnosed_diabetes'

CATS = ['gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status']

NUMS = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
        'triglycerides', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history']


BASE = [col for col in train.columns if col not in ['id', TARGET]]
ORIG = [] 

BASE = [c for c in BASE if c in train.columns and c in orig.columns]


for col in BASE:
    # --- MEAN ENCODING ---
    if col in orig.columns:
        mean_map = orig.groupby(col)[TARGET].mean().reset_index()
        new_mean_col_name = f"orig_mean_{col}"
        
        # Rename the target column to the new feature name for merging
        mean_map = mean_map.rename(columns={TARGET: new_mean_col_name})
        
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')
        ORIG.append(new_mean_col_name)

    if col in orig.columns:
        new_count_col_name = f"orig_count_{col}"
        count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
        
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')
        ORIG.append(new_count_col_name)

print(f'{len(ORIG)} Orig Features Created!!')
FEATURES = BASE + ORIG
print(f'{len(FEATURES)} Total Features.')


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


for col in CATS:
    if col in X.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')


train_data = X.copy()
train_data[TARGET] = y


%%time

save_path = 'saved_models'
time_limit = 5000

if os.path.exists(save_path):
    shutil.rmtree(save_path)

predictor = TabularPredictor(
    label=TARGET,
    problem_type='binary',
    eval_metric='roc_auc',
    path=save_path
)

predictor.fit(
    train_data=train_data,
    presets='best_quality',
    num_bag_folds=5,
    num_stack_levels=2,
    excluded_model_types=['KNN'],
    fit_weighted_ensemble=True,
    time_limit=time_limit,
    verbosity=3
)


leaderboard = predictor.leaderboard(silent=True)
print(leaderboard)


positive_class = predictor.positive_class  
y_pred_proba = predictor.predict_proba(X_test)[positive_class]

submission = pd.DataFrame({
    'id': submission.id,  
    'target': y_pred_proba  
})

submission.to_csv("predictions.csv", index=False)
print(f"Saved predictions for class '{positive_class}' to predictions.csv")

