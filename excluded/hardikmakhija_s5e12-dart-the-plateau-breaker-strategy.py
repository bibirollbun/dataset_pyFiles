import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata

# 1. LOAD DATA
PATH = "/kaggle/input/playground-series-s5e12"
train = pd.read_csv(f"{PATH}/train.csv")
test = pd.read_csv(f"{PATH}/test.csv")
y = train['diagnosed_diabetes']

def final_engineer(df):
    # Creating a 'Metabolic Risk Score'
    df['risk_score'] = (df['bmi'] * 0.4) + (df['systolic_bp'] * 0.3) + (df['cholesterol_total'] * 0.3)
    # Categorical encoding
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    return df

train_df = final_engineer(train).drop(['id', 'diagnosed_diabetes'], axis=1)
test_df = final_engineer(test).drop(['id'], axis=1)

# 2. THE DART MODEL (The Breakthrough Algorithm)
# 'boosting_type': 'dart' is the key here
params = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 48,
    'feature_fraction': 0.8,
    'drop_rate': 0.1,      # This "drops" trees to prevent overfitting
    'skip_drop': 0.5,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1
}

print("ðŸŽ¯ Training DART Model for the Breakthrough...")
dtrain = lgb.Dataset(train_df, label=y)
model = lgb.train(params, dtrain, num_boost_round=1200)

# 3. GENERATE PREDICTIONS
preds = model.predict(test_df)

# 4. SAVE SUBMISSION
pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': preds}).to_csv('submission_dart.csv', index=False)
print("âœ… DART Submission Ready!")

