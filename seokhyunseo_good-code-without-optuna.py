import sys
sys.path.append('/kaggle/input/iterativestratification')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.auto import tqdm
import catboost as cb
from sklearn.metrics import log_loss
from datetime import datetime

# Load data
train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')

# Drop ID
train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)

# Clean column names
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# Define numeric and categorical columns
num_cols = [i for i in train.columns.tolist() if i != 'Class']
cat_cols = 'EJ'
num_cols.remove(cat_cols)

# Encode categorical variable 'EJ'
encoder = LabelEncoder()
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# 1. 스케일 변환
from sklearn.preprocessing import PowerTransformer
train['DU_log'] = np.log1p(train['DU'])
train['DU_squared'] = train['DU'] ** 2
train['DU_inverse'] = 1 / (train['DU'] + 1e-6)
train['AB_squared'] = train['AB'] ** 2

test['DU_log'] = np.log1p(test['DU'])
test['DU_squared'] = test['DU'] ** 2
test['DU_inverse'] = 1 / (test['DU'] + 1e-6)
test['AB_squared'] = test['AB'] ** 2

train['BQ_log'] = np.log1p(train['BQ'])
test['BQ_log'] = np.log1p(test['BQ'])

num_cols = [i for i in train.columns.tolist() if i != 'Class']
cat_cols = 'EJ'
num_cols.remove(cat_cols)



# Adversarial Validation to align train-test distributions
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Train a simpler adversarial classifier
adv_params = {
    'iterations': 500,  # Reduced to prevent overfitting
    'learning_rate': 0.01,
    'depth': 4,
    'l2_leaf_reg': 15,  # Increased regularization
    'early_stopping_rounds': 200,
    'loss_function': 'Logloss',
    'random_seed': 42,
    'verbose': 0,
}
adv_model = cb.CatBoostClassifier(**adv_params)
adv_model.fit(
    combined[num_cols + [cat_cols]],
    combined['is_train'],
    verbose=0
)

# Compute adversarial weights and soften them
train['adv_weight'] = adv_model.predict_proba(train[num_cols + [cat_cols]])[:, 0]  # P(is_train)
train['adv_weight'] = 0.7 * train['adv_weight'] + 0.3 * 1.0  # Blend with uniform weights
train['adv_weight'] = np.clip(train['adv_weight'], 0.1, 10.0)  # Avoid extreme weights
train['adv_weight'] = train['adv_weight'] / train['adv_weight'].mean()  # Normalize

# Drop temporary columns
train = train.drop(['is_train'], axis=1)
test = test.drop(['is_train'], axis=1)

# Initialize arrays for OOF predictions and final test predictions
oof = np.zeros((len(train), 2))
final_preds = []




# Define CatBoost parameters
params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 4,
    'l2_leaf_reg': 10,
    'early_stopping_rounds': 500,
    'auto_class_weights': 'Balanced',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': 42,
    'verbose': 0,
}

# Hardness-based stratification
baseline_oof = np.zeros(len(train))
skf_baseline = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf_baseline.split(train, greeks.iloc[:, 1:-1]):
    X_train = train.loc[train_idx, num_cols + [cat_cols]]
    y_train = train.loc[train_idx, 'Class']
    X_val = train.loc[val_idx, num_cols + [cat_cols]]
    y_val = train.loc[val_idx, 'Class']
    
    baseline_model = cb.CatBoostClassifier(**params)
    baseline_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=train.loc[train_idx, 'adv_weight'],
        verbose=0
    )
    
    baseline_preds = baseline_model.predict_proba(X_val)[:, 1]
    hardness = np.where(
        ((y_val == 1) & (baseline_preds < 0.2)) | ((y_val == 0) & (baseline_preds > 0.8)),
        1,
        0
    )
    baseline_oof[val_idx] = hardness

# Main CV loop with 10 folds
skf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
multi_labels = pd.concat([train['Class'], pd.Series(baseline_oof, name='hardness')], axis=1)

best_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, multi_labels)):
    X_train = train.loc[train_idx, num_cols + [cat_cols]]
    y_train = train.loc[train_idx, 'Class']
    X_val = train.loc[val_idx, num_cols + [cat_cols]]
    y_val = train.loc[val_idx, 'Class']
    
    fold_models = []
    fold_scores = []
    
    for aa in range(3):
        params['random_seed'] += aa
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=train.loc[train_idx, 'adv_weight'],
            verbose=0
        )
        val_preds = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, val_preds)
        fold_models.append(model)
        fold_scores.append(score)
    print(aa, fold_scores)
    
    # Select top 2 models
    top_indices = np.argsort(fold_scores)[:2]
    for idx in top_indices:
        best_models.append(fold_models[idx])
    
    val_preds = np.mean([fold_models[idx].predict_proba(X_val) for idx in top_indices], axis=0)
    oof[val_idx, :] = val_preds
    
    test_preds = np.mean(
        [fold_models[idx].predict_proba(test[num_cols + [cat_cols]]) for idx in top_indices], axis=0
    )
    final_preds.append(test_preds)

# Print OOF log loss for debugging
print("OOF Log Loss:", log_loss(train['Class'], oof[:, 1]))

# Optimize reweighting alpha
def reweight_probs(probs, alpha=0.5):
    probs[:, 1] = probs[:, 1] ** alpha
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

best_alpha = 0.7
best_score = log_loss(train['Class'], oof[:, 1])
for alpha in np.linspace(0.3, 1.0, 20):
    reweighted_oof = reweight_probs(oof.copy(), alpha=alpha)
    score = log_loss(train['Class'], reweighted_oof[:, 1])
    if score < best_score:
        best_score = score
        best_alpha = alpha

# Average test predictions and reweight
final_probs = np.mean(final_preds, axis=0)
final_probs = reweight_probs(final_probs, alpha=best_alpha)

sample_submission = pd.read_csv(
    '/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
sample_submission[['class_0', 'class_1']] = final_probs
sample_submission.to_csv('submission.csv', index=False)



num_cols = [i for i in train.columns.tolist() if i != 'Class']
for a in range(len(best_models)):
    importances = best_models[a].get_feature_importance()
    top_features = [num_cols[i] for i in np.argsort(importances)[::-1][:10]]
    
    print(top_features)


train.BQ.plot()


import sys
sys.path.append('/kaggle/input/iterativestratification')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.auto import tqdm
import catboost as cb
from sklearn.metrics import log_loss
from datetime import datetime

# Load data
train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')

import matplotlib.pyplot as plt

numeric_columns = train.select_dtypes(include=['number']).columns





