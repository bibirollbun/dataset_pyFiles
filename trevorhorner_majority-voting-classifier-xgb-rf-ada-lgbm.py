# ─── Data Manipulation ─────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ─── Data Visualization ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Preprocessing & Feature Engineering ────────────────────────────────────
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

# ─── Model Selection & Pipeline ──────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold

# ─── Classifiers ─────────────────────────────────────────────────────────────
from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train['Personality'] = train['Personality'].map({'Extrovert': 0, 'Introvert': 1})


cat_c = ['Drained_after_socializing','Stage_fear']


def features(df):
    df['social_activity_score'] = (
        0.5 * df['Social_event_attendance'].fillna(0) +
        0.3 * df['Going_outside'].fillna(0) +
        0.2 * df['Post_frequency'].fillna(0)
    )
    df['friend_post_density'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1e-5)

    df['introversion_index'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)

    df['interactiveness'] = (
        df['Friends_circle_size'].fillna(0) +
        df['Social_event_attendance'].fillna(0) +
        df['Post_frequency'].fillna(0)
    )

    return df


train = features(train)
test = features(test)


def update(df):

    for col in cat_c:
        df[col] = df[col].fillna('Missing').astype('category')
    return df


train = update(train)
test = update(test)

train.head()


X = train.drop(columns=['Personality'])
y = train['Personality']



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss', 
    'tree_method': 'hist',
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'lambda': 1,
    'alpha': 0.5,
    'seed': 42
}


from sklearn.metrics import accuracy_score

fold_scores_XGB, test_preds_XGB = [], []

# Cross-validation training
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    X_train = X.loc[train_idx, cat_c].copy().reset_index(drop=True)
    X_valid = X.loc[valid_idx, cat_c].copy().reset_index(drop=True)

    y_train = y.loc[train_idx].reset_index(drop=True)
    y_valid = y.loc[valid_idx].reset_index(drop=True)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]

    for col in cat_c:
        X_train[col] = X_train[col].astype('category')
        X_valid[col] = X_valid[col].astype('category')

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    val_preds = bst.predict(dvalid)
    val_preds = np.round(val_preds).astype(int)
    acc = accuracy_score(y_valid, val_preds)

    fold_scores_XGB.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Final performance
print(f"\nMean Accuracy: {np.mean(fold_scores_XGB):.4f}")


X_test = test[cat_c].copy()
for col in cat_c:
    X_test[col] = X_test[col].astype('category')

dtest = xgb.DMatrix(X_test, enable_categorical=True)
test_preds_encoded = bst.predict(dtest)
test_preds_encoded = np.argmax(test_preds_encoded, axis=1) if test_preds_encoded.ndim > 1 else np.round(test_preds_encoded).astype(int)

# Map 0/1 back to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds = pd.Series(test_preds_encoded).map(label_map)

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else range(len(test)),
    "Personality": test_preds
})

submission.to_csv("submissionXGB8.csv", index=False)
print("✅ Submission file saved as 'submissionXGB8.csv'")


aboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_leaf=5),
    n_estimators=250,
    learning_rate=1.0,
    random_state=42
)

fold_scores_ADA, test_preds_ADA = [], []

# Cross-validation training
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"Fold {fold}")

    X_train = X.loc[train_idx, cat_c].copy().reset_index(drop=True)
    X_valid = X.loc[valid_idx, cat_c].copy().reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_valid = y.loc[valid_idx].reset_index(drop=True)

    # Convert categorical using one-hot encoding
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_valid_enc = pd.get_dummies(X_valid, drop_first=True)

    # Align columns between train and valid
    X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # Train AdaBoost
    aboost.fit(X_train_enc, y_train)
    
    val_preds = aboost.predict(X_valid_enc)
    acc = accuracy_score(y_valid, val_preds)
    fold_scores_ADA.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Final performance
print(f"\nMean Accuracy: {np.mean(fold_scores_ADA):.4f}")


X_test = test[cat_c].copy()
for col in cat_c:
    X_test[col] = X_test[col].astype('category')

X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align test columns with training columns
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

test_preds_encoded = aboost.predict(X_test_enc)

# Map 0/1 back to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds = pd.Series(test_preds_encoded).map(label_map)

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else range(len(test)),
    "Personality": test_preds
})

submission.to_csv("submission_ADA9.csv", index=False)
print("✅ Submission file saved as 'submission_ADA9.csv'")


rf_model = RandomForestClassifier(
    max_depth=5,
    n_estimators=1000,
    random_state=42
)


fold_scores_RF, test_preds_RF = [], []

# Cross-validation training
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"Fold {fold}")

    X_train = X.loc[train_idx, cat_c].copy().reset_index(drop=True)
    X_valid = X.loc[valid_idx, cat_c].copy().reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_valid = y.loc[valid_idx].reset_index(drop=True)

    # One-hot encode categorical features
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_valid_enc = pd.get_dummies(X_valid, drop_first=True)

    # Align columns between train and valid
    X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # Train Random Forest
    rf_model.fit(X_train_enc, y_train)

    val_preds = rf_model.predict(X_valid_enc)
    acc = accuracy_score(y_valid, val_preds)
    fold_scores_RF.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Final performance
print(f"\nMean Accuracy: {np.mean(fold_scores_RF):.4f}")



X_test = test[cat_c].copy()
for col in cat_c:
    X_test[col] = X_test[col].astype('category')

X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

test_preds_encoded = rf_model.predict(X_test_enc)

# Map 0/1 back to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds = pd.Series(test_preds_encoded).map(label_map)

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else range(len(test)),
    "Personality": test_preds
})

submission.to_csv("submission_RF17.csv", index=False)
print("✅ Submission file saved as 'submission_RF17.csv'")


lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 128,
    'max_depth': -1,
    'min_data_in_leaf': 10,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.95,
    'bagging_freq': 3,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'metric': 'auc',
    'verbose': -1
}


import lightgbm as lgb

fold_scores_LGB, test_preds_LGB = [], []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"Fold {fold}")

    X_train = X.loc[train_idx, cat_c].copy().reset_index(drop=True)
    X_valid = X.loc[valid_idx, cat_c].copy().reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_valid = y.loc[valid_idx].reset_index(drop=True)

    # One-hot encode categorical features
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_valid_enc = pd.get_dummies(X_valid, drop_first=True)

    # Align validation to training columns
    X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    dtrain = lgb.Dataset(X_train_enc, label=y_train)
    dvalid = lgb.Dataset(X_valid_enc, label=y_valid, reference=dtrain)

    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=1000,
    )

    val_preds = model.predict(X_valid_enc)
    val_preds_binary = (val_preds >= 0.5).astype(int)
    acc = accuracy_score(y_valid, val_preds_binary)
    fold_scores_LGB.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Final performance
print(f"\nMean Accuracy: {np.mean(fold_scores_LGB):.4f}")


X_test = test[cat_c].copy()
for col in cat_c:
    X_test[col] = X_test[col].astype('category')

X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

test_preds_prob = model.predict(X_test_enc)
test_preds_encoded = (test_preds_prob >= 0.5).astype(int)

# Map 0/1 back to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds = pd.Series(test_preds_encoded).map(label_map)

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else range(len(test)),
    "Personality": test_preds
})

submission.to_csv("submission_LGB8.csv", index=False)
print("✅ Submission file saved as 'submission_LGB8.csv'")


from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=42)),
        ('ada', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
            n_estimators=250, learning_rate=1.0, random_state=42)),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
            tree_method='hist', learning_rate=0.1, max_depth=3, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1, reg_alpha=0.5, random_state=42)),
        ('lgb', lgb.LGBMClassifier(
            objective='binary', boosting_type='gbdt', learning_rate=0.01, num_leaves=128,
            max_depth=-1, min_child_samples=10, feature_fraction=0.95, bagging_fraction=0.95,
            bagging_freq=3, reg_alpha=0.1, reg_lambda=0.1, metric='auc', verbose=-1))
    ],
    voting='hard',
    n_jobs=-1
)


# One-hot encode full data
X_full = pd.get_dummies(X[cat_c], drop_first=True)
X_test_enc = pd.get_dummies(test[cat_c], drop_first=True)

# Align columns
X_test_enc = X_test_enc.reindex(columns=X_full.columns, fill_value=0)


voting_clf.fit(X_full, y)

test_preds_encoded = voting_clf.predict(X_test_enc)

# Map back to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
test_preds = pd.Series(test_preds_encoded).map(label_map)

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else range(len(test)),
    "Personality": test_preds
})

submission.to_csv("submission_voting.csv", index=False)
print("✅ Submission file saved as 'submission_voting.csv'")

