import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


def engineer_features(df):
    df['bmi_waist_ratio'] = df['bmi'] / (df['waist_to_hip_ratio'] + 1e-5)
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    return df

train = engineer_features(train)
test = engineer_features(test)


target = 'diagnosed_diabetes'
features = [col for col in train.columns if col not in ['id', target]]
cat_features = [
    'gender', 'ethnicity', 'education_level', 'income_level', 
    'smoking_status', 'employment_status', 'family_history_diabetes', 
    'hypertension_history', 'cardiovascular_history'
]

X, y = train[features], train[target]
X_test = test[features]


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
test_preds = np.zeros(len(test))

for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
    X_t, X_v = X.iloc[t_idx], X.iloc[v_idx]
    y_t, y_v = y.iloc[t_idx], y.iloc[v_idx]


    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.01,   
        depth=4,              
        l2_leaf_reg=15,       
        eval_metric='AUC',
        cat_features=cat_features,
        early_stopping_rounds=200,
        verbose=1000
    )

    model.fit(X_t, y_t, eval_set=(X_v, y_v))
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f"Fold {fold+1} complete...")


# Create submission file
output = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_preds
})

output.to_csv('submission.csv', index=False)
print("Submission file saved!")

