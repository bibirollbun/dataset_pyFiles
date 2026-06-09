import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv').set_index("id")
orig_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# Rename temperature column
def rename_temperature_column(df):
    return df.rename(columns={'Temparature': 'Temperature'})

train = rename_temperature_column(train)
test = rename_temperature_column(test)
orig_data = rename_temperature_column(orig_data)


# Merge datasets
train = pd.concat([train, orig_data], ignore_index=True)


# Feature Encoding
cat_cols = train.select_dtypes(include=['object', 'category']).columns.drop('Fertilizer Name')
oe = OrdinalEncoder()
train[cat_cols] = oe.fit_transform(train[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])


# Target Encoding
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# Optimize dtypes
for df in [train, test]:
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int16')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')


# Prepare data
X_full = train.drop('Fertilizer Name', axis=1)
y_full = train['Fertilizer Name']
X_test = test


# Class weights for sample_weight
class_weights = np.bincount(y_full)
class_weights = class_weights.max() / class_weights
sample_weight_map = dict(enumerate(class_weights))
sample_weights_full = y_full.map(sample_weight_map)


# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Step 1: Initial training for pseudo-labeling
initial_model = XGBClassifier(
    max_depth=17, colsample_bytree=0.467, subsample=0.86, n_estimators=1000,
    learning_rate=0.03, gamma=0.26, max_delta_step=4, reg_alpha=2.7,
    reg_lambda=1.4, early_stopping_rounds=50, objective='multi:softprob',
    random_state=13, enable_categorical=True, tree_method='hist', device='cuda'
)

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
initial_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

# Predict test set
test_probs = initial_model.predict_proba(X_test)


# Pseudo-labeling
conf_threshold = 0.85
conf_mask = test_probs.max(axis=1) > conf_threshold
X_pseudo = X_test[conf_mask].copy()
y_pseudo = np.argmax(test_probs[conf_mask], axis=1)


# Merge real and pseudo-labeled data
X_combined = pd.concat([X_full, X_pseudo], ignore_index=True)
y_combined = pd.concat([y_full, pd.Series(y_pseudo)], ignore_index=True)

# Recalculate sample weights
sample_weights_combined = y_combined.map(sample_weight_map)

print(f"Added {len(X_pseudo)} pseudo-labeled samples")


# Step 2: CV Training
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

num_classes = y_full.nunique()
oof = np.zeros((len(y_full), num_classes))
pred_prob = np.zeros((len(X_test), num_classes))

# Keep track of real indices only
real_idx = np.arange(len(y_full))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_full, y_full)):
    print(f"\nFold {fold + 1}")

    # Training on combined data
    X_train_fold = pd.concat([X_full.iloc[train_idx], X_pseudo], ignore_index=True)
    y_train_fold = pd.concat([y_full.iloc[train_idx], pd.Series(y_pseudo)], ignore_index=True)
    sw_train_fold = y_train_fold.map(sample_weight_map)

    xgb_model = XGBClassifier(
        max_depth=17, colsample_bytree=0.467, subsample=0.86, n_estimators=1000,
        learning_rate=0.03, gamma=0.26, max_delta_step=4, reg_alpha=2.7,
        reg_lambda=1.4, early_stopping_rounds=50, objective='multi:softprob',
        random_state=13, enable_categorical=True, tree_method='hist', device='cuda'
    )

    xgb_model.fit(X_train_fold, y_train_fold, sample_weight=sw_train_fold,
                  eval_set=[(X_full.iloc[valid_idx], y_full.iloc[valid_idx])], verbose=0)

    # OOF predictions (only on real validation set)
    oof[valid_idx] = xgb_model.predict_proba(X_full.iloc[valid_idx])
    pred_prob += xgb_model.predict_proba(X_test) / FOLDS

    # Score MAP@3 only on real labels
    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score = mapk([[label] for label in y_full.iloc[valid_idx]], top_3_preds)
    print(f"Fold {fold + 1} MAP@3: {map3_score:.5f}")

    # Optional: Log loss
    loss = log_loss(y_full.iloc[valid_idx], oof[valid_idx])
    print(f"Fold {fold + 1} Log Loss: {loss:.5f}")


# Submission
top_k_indices = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_k_labels = le.inverse_transform(top_k_indices.ravel()).reshape(top_k_indices.shape)


submission = pd.DataFrame({
    'id': pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")['id'],
    'Fertilizer Name': [' '.join(row) for row in top_k_labels]
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved: submission.csv")


