import pandas as pd
import numpy as np

# Load the training dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Display first few rows
train.head()


train.info()


train.describe()


print(train.nunique().sort_values(ascending=False))


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_ids = test['id']


# Basic Feature Engineering
def create_features(df):
    df = df.copy()
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['M_T_mult'] = df['Moisture'] * df['Temparature']
    df['K_M_ratio'] = df['Potassium'] / (df['Moisture'] + 1e-5)
    df['Temp_bin'] = pd.qcut(df['Temparature'], 5, labels=False)
    return df

train = create_features(train)
test = create_features(test)


# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(train['Fertilizer Name'])

# Drop unused columns
train.drop(columns=['id', 'Fertilizer Name'], inplace=True)
test.drop(columns=['id'], inplace=True)


# Prepare cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage
oof_preds = np.zeros((len(train), len(le_target.classes_)))
test_preds = np.zeros((len(test), len(le_target.classes_)))
meta_features = np.zeros((len(train), len(le_target.classes_)))
meta_test = np.zeros((len(test), len(le_target.classes_)))


# Identify categorical columns by name or index
categorical_cols = ['Soil Type', 'Crop Type', 'Temp_bin']

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'task_type': 'GPU',         # â†� If you're using a GPU (P100 or T4)
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'verbose': 100
}

# Update the training block
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    cat = CatBoostClassifier(**cat_params)
    cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        cat_features=categorical_cols,  # ğŸ”¥ THIS IS THE FIX
        use_best_model=True
    )

    oof_preds[va_idx] = cat.predict_proba(X_va)
    test_preds += cat.predict_proba(test) / n_splits
    meta_features[va_idx] = oof_preds[va_idx]


from sklearn.linear_model import LogisticRegression

# Train meta-learner on out-of-fold predictions
meta_learner = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    random_state=42
)
meta_learner.fit(meta_features, y)


# Meta-test predictions
meta_test = meta_learner.predict_proba(test_preds)


# Combine predictions (simple arithmetic average)
final_preds = (test_preds + meta_test) / 2


# Prepare top-3 submission
top3 = np.argsort(-final_preds, axis=1)[:, :3]
top3_labels = le_target.inverse_transform(top3.ravel()).reshape(top3.shape)

submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved as 'submission.csv'")

