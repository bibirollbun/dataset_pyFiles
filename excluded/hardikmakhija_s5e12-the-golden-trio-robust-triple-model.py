import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata

# 1. LOAD & CLEAN
PATH = "/kaggle/input/playground-series-s5e12"
train = pd.read_csv(f"{PATH}/train.csv")
test = pd.read_csv(f"{PATH}/test.csv")
y = train['diagnosed_diabetes']

def final_polish(df):
    # High-impact medical interactions
    df['metabolic_index'] = (df['bmi'] * df['systolic_bp']) / 100
    df['age_risk'] = df['age'] * df['cholesterol_total']
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    return df

train_final = final_polish(train).drop(['id', 'diagnosed_diabetes'], axis=1)
test_final = final_polish(test).drop(['id'], axis=1)

# 2. THE TRIO (Training 3 diverse perspectives)
print("ğŸš€ Training LightGBM (The Speedster)...")
m1 = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63, verbose=-1, random_state=42)
m1.fit(train_final, y)
p1 = m1.predict_proba(test_final)[:, 1]

print("ğŸ”¥ Training XGBoost (The Specialist)...")
m2 = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=7, random_state=42)
m2.fit(train_final, y)
p2 = m2.predict_proba(test_final)[:, 1]

print("ğŸ�± Training CatBoost (The Robust One)...")
m3 = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=7, verbose=0, random_state=42)
m3.fit(train_final, y)
p3 = m3.predict_proba(test_final)[:, 1]

# 3. FINAL RANK BLENDING
# Giving more weight to LGBM and CatBoost as they usually generalize better
r1, r2, r3 = rankdata(p1), rankdata(p2), rankdata(p3)
final_ranks = (r1 * 0.40) + (r2 * 0.20) + (r3 * 0.40)
final_preds = final_ranks / final_ranks.max()

# 4. EXPORT
pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': final_preds}).to_csv('FINAL_SUBMISSION_S5E12.csv', index=False)
print("âœ… ALL MODELS COMPLETE. Submit 'FINAL_SUBMISSION_S5E12.csv'!")

