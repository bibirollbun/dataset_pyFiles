import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation


SEED = 42
N_SPLITS = 5


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)


train_df.head()


target_col = 'Fertilizer Name'
cat_features = ['Soil Type', 'Crop Type']


train_df[target_col].value_counts()


# # Label encoding - categorical features
# combined = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0)
# label_encoders = {}

# for col in cat_features:
#     le = LabelEncoder()
#     combined[col] = le.fit_transform(combined[col])
#     label_encoders[col] = le

# train_df[cat_features] = combined.iloc[:len(train_df)][cat_features]
# test_df[cat_features] = combined.iloc[:len(test_df)][cat_features]


# One-Hot Encoding - categorical features
combined = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0)
combined_encoded = pd.get_dummies(combined, columns=cat_features, drop_first=False)
train_df = pd.concat([combined_encoded.iloc[:len(train_df)], train_df[[target_col]].reset_index(drop=True)], axis=1)
test_df = combined_encoded.iloc[len(train_df):].reset_index(drop=True)


# Label encoding - target feature
target_le = LabelEncoder()
train_df[target_col] = target_le.fit_transform(train_df[target_col])
train_df = pd.concat([combined_encoded.iloc[:len(train_df)], train_df[[target_col]].reset_index(drop=True)], axis=1)
test_df = combined_encoded.iloc[len(train_df):].reset_index(drop=True)


# Define features and target
features = train_df.drop(columns=[target_col]).columns

X = train_df[features]
y = train_df[target_col]
X_test = test_df[features]


# Define a MAP@K function
def mapk(y_true, y_pred_proba, k=3):
    topk = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]
    y_true = np.array(y_true).reshape(-1, 1)
    scores = (topk == y_true).astype(int)
    return np.mean([score[:k].argmax()+1 if score[:k].sum() > 0 else 0 for score in scores]) / k


# Cross validation - StratifiedKFold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((len(X), len(np.unique(y))))
test_preds = np.zeros((len(X_test), len(np.unique(y))))
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"============== Fold {fold+1} ==============")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Training
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=6,
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=SEED,
        class_weight='balanced',
        importance_type='gain'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=0)
        ]
    )

    # Make predictions on the val set
    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = val_probs

    # Make predictions on the test set
    test_preds += model.predict_proba(X_test) / skf.n_splits

    # Calculate MAP@3 score
    score = mapk(y_val, val_probs, k=3)
    scores.append(score)
    print("")
    print(f"------------>  MAP@3 = {score:.4f}")
    print("")


print(f"\nMean MAP@3 across folds: {np.mean(scores):.4f}")


top3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels = target_le.inverse_transform(top3.ravel()).reshape(top3.shape)
final_preds = [' '.join(row) for row in top3_labels]

submission = test_df[['id']].copy()
submission['Fertilizer Name'] = final_preds
submission.to_csv("submission.csv", index=False)
print(submission.head())







