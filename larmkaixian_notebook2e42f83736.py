# data cleanup
import numpy as np
import pandas as pd
df = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
df = df.drop_duplicates()
df


df.isnull().sum()


print(df.info())


print(df.head())


def fet(dataframe):
    df = dataframe.copy()
    df['risk_score'] = 0.0
    # Parameter 1: Age, 30-44 mid risk, 45+ high risk (1 arg)
    df.loc[((df['age'] >= 30)&(df['age']<45)), 'risk_score'] += 0.5
    df.loc[df['age'] >= 45, 'risk_score'] += 1.0

    # Parameter 2: Smoking (1 arg)
    df.loc[df['smoking_status'] == 'Current', 'risk_score'] += 1.0

    # Parameter 3: Histories (3 args)
    df.loc[df['family_history_diabetes'] == 1, 'risk_score'] += 1.0
    df.loc[df['hypertension_history'] == 1, 'risk_score'] += 1.0
    df.loc[df['cardiovascular_history'] == 1, 'risk_score'] += 1.0

    # Parameter 4: Waist-to-hip (1 arg)
    df.loc[((df['gender'] == 'Female') & ((df['waist_to_hip_ratio'] >= 0.85) & df['waist_to_hip_ratio'] < 0.90)), 'risk_score'] += 0.5
    df.loc[((df['gender'] == 'Female') & (df['waist_to_hip_ratio'] >= 0.90)), 'risk_score'] += 1.0

    df.loc[((df['gender'] == 'Male') & ((df['waist_to_hip_ratio'] >= 0.90) & df['waist_to_hip_ratio'] < 1.00)), 'risk_score'] += 0.5
    df.loc[((df['gender'] == 'Male') & (df['waist_to_hip_ratio'] >= 1.00)), 'risk_score'] += 1.0

    # Parameter 5: Fatty liver? (1 arg)
    df['tg_hdl_ratio'] = df['triglycerides']/df['hdl_cholesterol']
    df.loc[((df['tg_hdl_ratio'] >= 3.0) & (df['tg_hdl_ratio'] < 4.0)), 'risk_score'] += 0.5
    df.loc[df['tg_hdl_ratio'] >= 4.0, 'risk_score'] += 1.0

    # Parameter 6: HBP (1 arg)
    df['map'] = df['diastolic_bp'] + ( 1/3 * (df['systolic_bp'] - df['diastolic_bp']))
    df.loc[((df['map'] >= 92) & (df['map'] < 96)), 'risk_score'] += 0.5
    df.loc[df['map'] >= 96, 'risk_score'] += 1.0

    # Parameter 7: Cholesterol (1 arg)
    df.loc[df['cholesterol_total']>= 240, 'risk_score'] += 1.0
    df.loc[((df['cholesterol_total'] >= 200) & (df['cholesterol_total']<240)), 'risk_score'] += 0.5

    # Lifestyle penalty
    df['lifestyle_penalty'] = 0.0
    # Sedentary ratio
    activity_hours = df['physical_activity_minutes_per_week'] / 60
    df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (activity_hours + 0.1)

    # diet score
    df.loc[(df['diet_score'] < 3.0), 'lifestyle_penalty'] += 2.0 # higher penalty
    df.loc[((df['diet_score'] >= 3.0) & (df['diet_score'] < 5.0)), 'lifestyle_penalty'] += 1.0
    df.loc[((df['diet_score'] >= 5.0) & (df['diet_score'] < 7.0)), 'lifestyle_penalty'] += 0.5
    df.loc[(df['diet_score'] >= 7.0), 'lifestyle_penalty'] += 0 # no penalty
    # bad sleep scheldule
    bad_sleep_mask = (df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)
    df.loc[bad_sleep_mask, 'lifestyle_penalty'] += 1.0
    # assumption: > 14 drinks/week is bad
    df.loc[df['alcohol_consumption_per_week'] > 14, 'lifestyle_penalty'] += 1.0
    # physical activity
    df.loc[df['physical_activity_minutes_per_week'] < 150, 'lifestyle_penalty'] += 1.0

    avg_risk_score = df['risk_score'].mean()
    avg_health_penalty = df['lifestyle_penalty'].mean()

    # good health lifestyle, bad score, genetic problem
    df['genetic_suspect'] = 0
    genetic = (df['lifestyle_penalty'] < avg_health_penalty) & (df['risk_score'] > avg_risk_score)
    df.loc[genetic, 'genetic_suspect'] = 1

    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
    
    for col in cat_cols:
        if col in df.columns:
            # "Male" into 1, "Female" into 0 
            df[col] = df[col].astype('category').cat.codes

    
    return df
    


train_raw = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
test_raw = pd.read_csv(r"/kaggle/input/playground-series-s5e12/test.csv")


train_raw = train_raw.drop_duplicates()
test_raw = test_raw.drop_duplicates()

train_df = fet(train_raw)
test_df = fet(test_raw)

print("Train Columns:", train_df.head())
print("Test Columns:", test_df.head())


# Part 3: Train
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

feature_set = [
    'risk_score', 'lifestyle_penalty', 'sedentary_ratio', 'genetic_suspect','map', 'tg_hdl_ratio', 'age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides','gender', 'ethnicity', 'education_level', 'smoking_status', 'family_history_diabetes'
]
X = train_df[feature_set]
Y = train_df['diagnosed_diabetes']

#take note of diabetic
weight = (Y == 0).sum()/ (Y==1).sum()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = xgb.XGBClassifier(
    n_estimators=10000,
    learning_rate=0.005,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=weight,
    eval_metric='auc',
    early_stopping_rounds=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, Y_train, 
    eval_set=[(X_val, Y_val)], 
    verbose=False
)

val_preds = model.predict_proba(X_val)[:, 1]
print(f"Validation AUC: {roc_auc_score(Y_val, val_preds):.5f}")

test_preds = model.predict_proba(test_df[feature_set])[:, 1]

# 2. Create Submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Done!")

